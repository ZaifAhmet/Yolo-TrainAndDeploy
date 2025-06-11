import os
import cv2
import time
import threading
import base64 # Kareleri kodlamak/çözmek için
import numpy as np
from flask import Flask, render_template, Response, request, jsonify # jsonify: Web API'leri için JSON yanıtları oluşturmakta kullanılır
from ultralytics import YOLO

# --- Ayarlar ---
MODEL_PATH = 'best_trained_model.pt'  # Modelinizin yolu
CONFIDENCE_THRESHOLD = 0.3
# SOURCE_TYPE ve SOURCE_INPUT artık doğrudan kullanılmayacak,
# çünkü kaynak tarayıcıdan gelecek.
# DISPLAY_RESOLUTION web tarafında ayarlanabilir veya burada zorlanabilir.
# İsteğe bağlı: Sunucuda işlenen kareyi yeniden boyutlandırmak için
FORCE_SERVER_RESIZE = True # Tarayıcıdan geleni işledikten sonra yeniden boyutlandır
SERVER_RES_W, SERVER_RES_H = 416, 416

# --- Flask Uygulamasını Başlat ---
app = Flask(__name__) # Flask uygulamasını başlatır

# --- Model ve Diğer Global Değişkenler ---
print(f"Loading YOLO model from {MODEL_PATH}...")
try:
    # ÖNCE torch'u import et (Bazı CUDA başlatma sıralamaları için önemli olabilir)
    import torch

    # Sonra cihazı belirle
    device_to_use = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Checking CUDA availability: {torch.cuda.is_available()}")
    print(f"Attempting to load model on device: {device_to_use}")

    # Modeli yükle
    model = YOLO(MODEL_PATH, task='detect')
    # Modeli belirtilen cihaza taşı
    model.to(device_to_use)

    labels = model.names # Modelin sınıf etiketlerini alır
    # model.device ile teyit et (YOLO objesi doğrudan device attribute'u sağlamayabilir,
    # ama modelin parametrelerinin cihazını kontrol edebiliriz)
    # Eğer model.parameters() boş değilse ilk parametrenin cihazına bakalım
    try:
        model_device = next(model.parameters()).device
        print(f"Model loaded successfully. Parameters are on device: {model_device}")
    except StopIteration:
        print("Model loaded successfully (no parameters found to check device).")
    except AttributeError:
        print("Model loaded successfully (could not check parameter device).")


except Exception as e:
    print(f"Error loading YOLO model: {e}")
    # Hata durumunda PyTorch CUDA durumunu tekrar kontrol edelim
    # Hata mesajında torch yoksa tekrar import etmeye gerek yok ama garanti olsun
    try:
        import torch
        print(f"Re-checking CUDA availability during exception: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Current CUDA device: {torch.cuda.current_device()}")
                print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    except ImportError:
        print("Could not import torch during exception handling.")
    except Exception as inner_e:
        print(f"Error during CUDA check in exception handler: {inner_e}")
    exit() # Model yüklenemezse uygulamayı sonlandır

# Global değişkenler
latest_received_frame = None # Tarayıcıdan gelen son kareyi saklamak için
output_frame = None         # İşlenmiş ve MJPEG için kullanılacak kare
lock = threading.Lock()     # Thread güvenliği için (global değişkenlere erişirken)

# Bounding box renkleri (farklı sınıflar için)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# --- Arka Plan İşleme Thread'i ---
def process_frames():
    global latest_received_frame, output_frame, lock, model, labels, bbox_colors, CONFIDENCE_THRESHOLD, FORCE_SERVER_RESIZE, SERVER_RES_W, SERVER_RES_H

    frame_rate_buffer = [] # FPS hesaplaması için son kare hızlarını tutar
    fps_avg_len = 30 # FPS ortalaması için tampon boyutu (kaç kare üzerinden ortalama alınacağı)
    frame_counter = 0 # Alınan kare sayacı (kare atlama için kullanılır)
    # Her N'inci kareyi işle (örneğin 4 ise her 4 karede bir işle). Performans için ayarlanabilir.
    # 1 değeri her kareyi işler.
    PROCESS_EVERY_N_FRAMES = 4

    # Modelin hangi cihazda olduğunu alalım (başlangıçta belirlenmiş olmalı)
    try:
        current_device = next(model.parameters()).device
        print(f"Processing thread will use device: {current_device}")
    except Exception:
        print("Could not determine model device for processing thread, assuming CPU or pre-set.")
        current_device = 'cuda' if torch.cuda.is_available() else 'cpu' # Güvenlik için tekrar kontrol et

    while True:
        t_start = time.perf_counter() # Kare işleme başlangıç zamanı
        frame_to_process = None
        new_frame_received = False

        with lock: # Thread güvenli erişim
            if latest_received_frame is not None:
                frame_to_process = latest_received_frame.copy() # İşlenecek kareyi kopyala
                latest_received_frame = None # İşlenmek üzere alındı, sıfırla ki tekrar işlenmesin
                new_frame_received = True

        if not new_frame_received:
            time.sleep(0.01) # Yeni kare yoksa çok kısa bekle ve döngüye devam et
            continue

        # --- Kare Atlama Mantığı (PROCESS_EVERY_N_FRAMES > 1 ise aktif) ---
        frame_counter += 1
        if PROCESS_EVERY_N_FRAMES > 1 and frame_counter % PROCESS_EVERY_N_FRAMES != 0:
           # output_frame'i en son başarılı işlenmiş kare olarak tutmaya devam etmesi için
           # buraya `output_frame`'a son işlenmiş kareyi atama mantığı eklenebilir
           # ya da basitçe atlayıp bir sonraki işlenecek kareyi bekleyebilir.
           # Şimdilik, atlanan kareler için MJPEG stream'i son başarılı kareyi göstermeye devam eder.
           time.sleep(0.01) # CPU'yu yormamak için kısa bir bekleme
           continue # Bu kareyi atla, bir sonraki iterasyona geç

        display_frame = frame_to_process # İşlenecek kareyi (veya kopyasını) al

        # --- YOLO Inference ---
        try:
            # Modeli GPU'da çalıştırıyorsak, frame'i de GPU'ya göndermek daha verimli olabilir
            # ancak Ultralytics kütüphanesi genellikle bunu otomatik yapar.
            # Explicit device ataması genellikle model yüklemede yeterlidir.
            # `imgsz` parametresi, modelin eğitildiği veya en iyi performansı verdiği boyuta ayarlanmalı.
            results = model(display_frame, imgsz=SERVER_RES_W, verbose=False, conf=CONFIDENCE_THRESHOLD)
        except Exception as e:
            print(f"Error during model inference: {e}")
            continue # Hata olursa bu frame'i atla ve bir sonrakini bekle

        detections = results[0].boxes # İlk resimdeki tespit edilen kutuları al
        object_count = 0 # Nesne sayacını her işlenen kare başında sıfırla

        # --- Tespitleri Çiz ---
        if len(detections) > 0: # Eğer tespit varsa çizim yap
             # `.cpu()` çağrıları GPU kullanılıyorsa gereklidir, veriyi CPU'ya taşır.
             # `.numpy()` ile de NumPy dizisine çeviririz.
             det_data = detections.data.cpu().numpy() # Tüm veriyi bir kerede alıp numpy'a çevir

             for i in range(len(det_data)):
                 box = det_data[i][:4] # İlk 4 değer: xmin, ymin, xmax, ymax (köşe koordinatları)
                 conf = det_data[i][4] # Tespitin güven skoru
                 classidx = int(det_data[i][5]) # Tespit edilen sınıfın indeksi

                 xmin, ymin, xmax, ymax = map(int, box) # Kutu koordinatlarını integer'a çevir

                 classname = labels.get(classidx, f"ID:{classidx}") # Sınıf adını etiketlerden al
                 color = bbox_colors[classidx % len(bbox_colors)] # Sınıfa göre renk seç

                 # Dikdörtgeni çiz
                 cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 2)

                 # Etiketi hazırla ve çiz (sınıf adı ve güven skoru)
                 label = f'{classname}: {int(conf*100)}%'
                 labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                 label_ymin = max(ymin, labelSize[1] + 10) # Etiketin taşmaması için y pozisyonunu ayarla
                 # Etiket için arka plan rengi
                 cv2.rectangle(display_frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                 # Etiket metni (siyah renkte)
                 cv2.putText(display_frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                 object_count += 1 # Tespit edilen nesne sayısını artır


        # --- FPS Hesaplama ---
        t_stop = time.perf_counter() # Kare işleme bitiş zamanı
        elapsed_time = t_stop - t_start # Kareyi işleme süresi
        frame_rate_calc = 1.0 / elapsed_time if elapsed_time > 0 else 0 # Saniyedeki kare sayısı (FPS)
        frame_rate_buffer.append(frame_rate_calc) # Hesaplanan FPS'i tampona ekle
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0) # Eski FPS değerini buffer'dan çıkar (kayan ortalama için)
        # Buffer boş değilse ortalamayı hesapla
        avg_frame_rate = np.mean(frame_rate_buffer) if frame_rate_buffer else 0

        # --- Bilgileri Frame Üzerine Yaz ---
        fps_text = f'FPS: {avg_frame_rate:.1f}' # FPS metni (virgülden sonra 1 basamak)
        obj_text = f'Objects: {object_count}'   # Nesne sayısı metni

        # Metinleri sol üste yazdır (Turkuaz (Cyan) renk, siyah çerçeve ile daha okunaklı olabilir)
        # cv2.LINE_AA daha yumuşak çizgiler sağlar.
        cv2.putText(display_frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, obj_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # --- Sunucuda Yeniden Boyutlandırma (isteğe bağlı) ---
        # Tarayıcıdan gelen kare işlendikten sonra, MJPEG stream için yeniden boyutlandırılabilir.
        if FORCE_SERVER_RESIZE:
            display_frame = cv2.resize(display_frame, (SERVER_RES_W, SERVER_RES_H))

        # --- İşlenmiş Frame'i Global Değişkene Kaydet (Thread Güvenli) ---
        with lock:
            output_frame = display_frame.copy() # MJPEG stream'i için kopyasını sakla


# --- MJPEG Stream Üretici Fonksiyonu ---
# Bu fonksiyon, işlenmiş `output_frame`'i kullanarak MJPEG video akışı üretir.
def generate_mjpeg():
    global output_frame, lock
    while True:
        frame_to_encode = None
        with lock: # Thread güvenli erişim
            if output_frame is not None:
                frame_to_encode = output_frame.copy() # Kodlanacak kareyi kopyala

        if frame_to_encode is not None:
            # Kareyi JPEG formatında kodla
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode)
            if not flag: # Kodlama başarısız olursa atla
                continue
            # MJPEG formatında yield et (parça parça gönder)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encodedImage) + b'\r\n')
        else:
            # İşlenmiş kare yoksa (örneğin, başlangıçta veya bir hata sonrası) kısa bir süre bekle
            time.sleep(0.1)


# --- Yeni Endpoint: Tarayıcıdan Kare Almak İçin ---
@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global latest_received_frame, lock
    try:
        data = request.get_json() # Gelen JSON verisini al
        if not data or 'image' not in data:
            return jsonify({"error": "Missing image data"}), 400 # 'image' alanı yoksa hata döndür

        # Base64 string'i al (örn: "data:image/jpeg;base64,/9j/...")
        img_b64 = data['image']

        # Veri URI başlığını kaldır (varsa) (örn: "data:image/jpeg;base64,")
        if ',' in img_b64:
            header, img_b64_data = img_b64.split(',', 1)
        else:
            img_b64_data = img_b64 # Başlık yoksa, tamamı veri

        # Base64'ü çöz ve OpenCV formatına getir (byte dizisi -> NumPy dizisi -> OpenCV karesi)
        img_bytes = base64.b64decode(img_b64_data)
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR) # Renkli olarak oku

        if frame is None: # Görüntü çözülemezse hata döndür
             return jsonify({"error": "Could not decode image"}), 400

        # Gelen kareyi global değişkene ata (thread-safe)
        with lock:
            latest_received_frame = frame

        return jsonify({"status": "ok", "message": "Frame received"}), 200 # Başarılı yanıt

    except base64.binascii.Error: # Geçersiz Base64 string hatası
         return jsonify({"error": "Invalid base64 string"}), 400
    except Exception as e:
        print(f"Error in /upload_frame: {e}")
        return jsonify({"error": "Internal server error"}), 500


# --- Flask Rotaları ---
@app.route("/")
def index():
    # Ana sayfayı (index_phone.html) render et
    return render_template("index_phone.html")

@app.route("/video_feed")
def video_feed():
    # İşlenmiş kareleri MJPEG olarak stream et
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Ana Çalıştırma Bloğu ---
if __name__ == '__main__':
    # Arka plan kare işleme thread'ini başlat
    print("Starting background frame processing thread...")
    # `daemon=True` ile ana thread sonlandığında bu thread de otomatik sonlanır.
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    processing_thread.start()
    print("Background thread started.")

    # --- HTTPS için Sertifika Ayarı ---
    # Tarayıcıların kamera erişimi için genellikle HTTPS gerekir.
    # Önce cert.pem (sertifika) ve key.pem (özel anahtar) dosyalarını oluşturmanız GEREKİR!
    # Örneğin terminalde (Linux/macOS için, Windows'ta OpenSSL kurulumu gerekebilir):
    # openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365 -subj "/CN=localhost"
    # Bu komut, 'localhost' için 365 gün geçerli kendinden imzalı bir sertifika oluşturur.
    # Bu dosyaların app.py ile aynı dizinde olduğundan emin olun veya doğru yolu belirtin.
    try:
        # Sertifika ve anahtar dosyalarının yollarını içeren bir tuple
        context = ('cert.pem', 'key.pem')
        print("Starting Flask server with HTTPS on https://0.0.0.0:5000")
        # Flask'ı HTTPS ile çalıştır
        # `threaded=True` Flask'ın birden fazla isteği aynı anda işlemesini sağlar.
        # `use_reloader=False` genellikle daemon thread'lerle daha uyumludur.
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False, ssl_context=context)
    except FileNotFoundError:
         print("*"*40)
         print("HATA: SSL sertifika dosyaları (cert.pem, key.pem) bulunamadı!")
         print("HTTPS'i etkinleştirmek ve kamera erişimini sağlamak için önce bu dosyaları oluşturun.")
         print("Örnek OpenSSL komutu yukarıdaki yorum satırlarında belirtilmiştir.")
         print("Alternatif olarak, geliştirme için ngrok gibi bir araçla HTTP'yi HTTPS'e tünelleyebilirsiniz.")
         print("*"*40)
         # Sertifikalar olmadan HTTP olarak başlatmayı denemek için aşağıdaki satırların yorumunu kaldırabilirsiniz
         # ancak tarayıcı kamera erişimine izin vermeyebilir.
         # print("Attempting to start Flask server with HTTP on http://0.0.0.0:5000 (CAMERA MIGHT NOT WORK)")
         # app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)