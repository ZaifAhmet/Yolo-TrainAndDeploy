# Derin Öğrenme Tabanlı Sebze/Meyve Tanıma Web Uygulaması

Bu proje, Python, YOLO (You Only Look Once), Flask ve modern web teknolojileri (HTML, CSS, JavaScript) kullanılarak geliştirilmiş, telefon kamerası aracılığıyla gerçek zamanlı sebze ve meyve tanıma yapabilen bir web uygulamasıdır. Kullanıcı, telefonunun kamerasını kullanarak çevresindeki sebze ve meyveleri sisteme gösterir; sistem bu görüntüleri işleyerek nesneleri tanır ve sonuçları ekranda canlı olarak gösterir.

Proje, özel bir veri kümesi üzerinde eğitilmiş bir YOLOv8 modelini temel alır ve bu modelin tahminlerini etkileşimli bir web arayüzü üzerinden sunar.

## Temel Özellikler

*   **Gerçek Zamanlı Nesne Tespiti:** Telefon kamerasından alınan görüntüler üzerinde canlı olarak sebze/meyve tespiti.
*   **Özel YOLO Modeli:** Projeye özel olarak toplanmış ve etiketlenmiş bir veri kümesi üzerinde eğitilmiş YOLO modeli.
*   **Etkileşimli Web Arayüzü:**
    *   Kullanıcının telefon kamerasını başlatıp durdurabilmesi.
    *   İşlenmiş video akışının (tespit edilen nesneler ve etiketleriyle birlikte) canlı gösterimi.
    *   Durum mesajları ile kullanıcıya geri bildirim.
*   **Arka Plan İşleme:** Flask sunucusu, gelen kamera karelerini ayrı bir thread'de işleyerek kullanıcı arayüzünün takılmasını engeller.
*   **MJPEG Video Akışı:** İşlenmiş görüntülerin verimli bir şekilde tarayıcıya aktarılması.
*   **Model Eğitim Süreci:** Veri toplama, hazırlama, YOLO modeli eğitimi, doğrulaması ve test süreçlerini içeren kapsamlı bir Jupyter Notebook.

## Kullanılan Teknolojiler

*   **Backend & Model:**
    *   Python 3.x
    *   Flask (Web sunucusu ve API endpoint'leri için)
    *   Ultralytics YOLO (Nesne tespiti modeli ve eğitim kütüphanesi)
    *   OpenCV (Görüntü işleme, kare manipülasyonu)
    *   NumPy (Sayısal işlemler ve veri manipülasyonu)
    *   PyTorch (Ultralytics tarafından kullanılan derin öğrenme kütüphanesi)
*   **Frontend:**
    *   HTML5
    *   CSS3
    *   JavaScript (ES6+)
*   **Model Eğitimi & Veri Hazırlığı:**
    *   Jupyter Notebook
    *   PyYAML (YAML konfigürasyon dosyaları için)
    *   Roboflow (Veri kümesi yönetimi ve indirme için - notebook'ta kullanılmış)
*   **Diğer:**
    *   Base64 (Görüntü verisinin kodlanması/çözülmesi)
    *   Threading (Eş zamanlı işlemler için)
    *   HTTPS (Güvenli kamera erişimi için OpenSSL ile oluşturulmuş sertifikalar)

## Proje Yapısı

Aşağıda projenin temel dosya ve klasör yapısı gösterilmiştir:

```text
├── app.py                         # Flask backend sunucusu (model yükleme, API, video işleme)
├── index_phone.html               # Frontend HTML sayfası (kamera erişimi, sunucuya veri gönderme, sonuçları gösterme)
├── CombinedVegetables&Fruits.ipynb # YOLO modeli eğitim, doğrulama ve test notebook'u
├── best_trained_model.pt          # Eğitilmiş ve uygulamada kullanılan YOLO modeli ağırlık dosyası (Notebook çıktısı)
├── cert.pem                       # HTTPS için SSL sertifikası (oluşturulmalı)
├── key.pem                        # HTTPS için SSL özel anahtarı (oluşturulmalı)
```

## Kurulum ve Çalıştırma

### 1. Ön Gereksinimler

*   Python 3.8 veya üzeri
*   pip (Python paket yöneticisi)
*   OpenSSL (SSL sertifikaları oluşturmak için, Linux/macOS'ta genellikle yüklüdür, Windows için kurulum gerekebilir)
*   (Önerilen) Bir sanal ortam (virtual environment) yöneticisi (örn: `venv`, `conda`)

### 2. Kurulum Adımları

1.  **Repoyu Klonlayın:**
    ```bash
    git clone https://github.com/ZaifAhmet/Yolo-TrainAndDeploy.git
    cd Yolo-TrainAndDeploy
    ```

2.  **(Önerilen) Sanal Ortam Oluşturun ve Aktive Edin:**
    ```bash
    python -m venv venv
    # Windows için:
    # venv\Scripts\activate
    # Linux/macOS için:
    # source venv/bin/activate
    ```

3.  **Gerekli Kütüphaneleri Yükleyin:**
    *   Proje ana dizininde aşağıdaki içeriğe sahip bir `requirements.txt` dosyası oluşturun:
        ```txt
        Flask
        opencv-python
        numpy
        ultralytics
        # torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX (PyTorch CUDA versiyonunuza göre ayarlayın ya da CPU için sadece torch)
        # Ultralytics genellikle PyTorch'u doğru bir şekilde kurar.
        ```
    *   Ardından yükleyin:
        ```bash
        pip install -r requirements.txt
        ```
        *Not: PyTorch ve CUDA kurulumu için [PyTorch resmi web sitesindeki](https://pytorch.org/get-started/locally/) talimatları izlemeniz en sağlıklısıdır. Ultralytics genellikle PyTorch'u kendisiyle uyumlu bir şekilde kurmaya çalışır.*

4.  **Eğitilmiş Model Dosyasını Edinin:**
    *   `CombinedVegetables&Fruits.ipynb` notebook'unu çalıştırarak kendi `best.pt` modelinizi eğitin ve bu dosyayı projenin ana dizinine `best_trained_model.pt` adıyla kaydedin.
    *   Ya da eğer önceden eğitilmiş bir modeliniz varsa, onu bu isimle ana dizine kopyalayın.

5.  **SSL Sertifikalarını Oluşturun (HTTPS için):**
    Tarayıcıların kamera erişimine izin vermesi için web sunucusunun HTTPS üzerinden çalışması gerekir. Proje ana dizininde terminali açın ve aşağıdaki komutu çalıştırın (OpenSSL gereklidir):
    ```bash
    openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365 -subj "/CN=localhost"
    ```
    Bu komut, 365 gün geçerli `cert.pem` (sertifika) ve `key.pem` (özel anahtar) dosyalarını oluşturacaktır.

### 3. Uygulamayı Çalıştırma

1.  Proje ana dizinindeyken Flask sunucusunu başlatın:
    ```bash
    python app.py
    ```
    Sunucu `https://0.0.0.0:5000` veya `https://localhost:5000` adresinde çalışmaya başlayacaktır.

2.  Telefonunuzdan (veya HTTPS destekleyen bir masaüstü tarayıcıdan) `https://<BILGISAYARINIZIN_YEREL_IP_ADRESI>:5000` adresine gidin.
    *   Bilgisayarınızın yerel IP adresini öğrenmek için Windows'ta `ipconfig`, Linux/macOS'ta `ifconfig` veya `ip addr` komutlarını kullanabilirsiniz.
    *   Telefonunuzun ve bilgisayarınızın aynı ağda olduğundan emin olun.
    *   Tarayıcı, kendinden imzalı sertifika nedeniyle bir güvenlik uyarısı verebilir. "Yine de devam et" veya "Riski kabul et" gibi bir seçenekle ilerleyin.

3.  Tarayıcı kamera erişim izni istediğinde izin verin.

4.  "Start Camera" butonuna tıklayarak tanıma işlemini başlatın.

## İş Akışı (Workflow)

1.  Kullanıcı `index_phone.html` sayfasını ziyaret eder.
2.  "Start Camera" butonuna tıklandığında, JavaScript telefon kamerasından görüntü akışını alır.
3.  Belirli aralıklarla (`UPLOAD_INTERVAL_MS`), JavaScript o anki kamera karesini yakalar, Base64 formatında JPEG'e dönüştürür.
4.  Bu Base64 veri, `fetch` API ile `app.py`'deki `/upload_frame` endpoint'ine POST isteği olarak gönderilir.
5.  `app.py` sunucusu, gelen Base64 veriyi çözer ve OpenCV karesine dönüştürür. Bu kare `latest_received_frame` global değişkenine atanır.
6.  Ayrı bir arka plan thread'inde çalışan `process_frames` fonksiyonu, `latest_received_frame`'i alır.
7.  Bu kare üzerinde YOLO modeli ile nesne tespiti yapılır.
8.  Tespit edilen nesnelerin etrafına sınırlayıcı kutular, sınıf isimleri ve güven skorları çizilir. FPS ve nesne sayısı gibi bilgiler de kareye eklenir.
9.  İşlenmiş bu kare, `output_frame` global değişkenine atanır.
10. `index_phone.html` sayfasındaki `<img>` elementi, `app.py`'deki `/video_feed` endpoint'inden MJPEG formatında canlı video akışını gösterir. Bu akış, `generate_mjpeg` fonksiyonu tarafından `output_frame` kullanılarak üretilir.

## Model Eğitimi (`CombinedVegetables&Fruits.ipynb`)

Bu Jupyter Notebook, projenin kalbi olan YOLO modelinin eğitim sürecini detaylı bir şekilde içerir:

1.  **Veri Hazırlığı:**
    *   Roboflow gibi bir platformdan veya kendi topladığınız veri kümesinin indirilip çıkarılması.
    *   Veri kümesinin `train`, `valid`, `test` olarak yeniden dengelenmesi ve YOLO formatına uygun klasör yapısının (`images`, `labels`) ve `data.yaml` dosyasının oluşturulması. Bu adım, modelin farklı veri dağılımlarına karşı daha robust olmasını sağlar.
2.  **Model Seçimi ve Kurulum:** Ultralytics kütüphanesi kurulur ve önceden eğitilmiş bir YOLO (örn: `yolov8n.pt`) modeli temel alınır.
3.  **Eğitim:** `model.train()` fonksiyonu kullanılarak hazırlanan veri kümesi üzerinde model eğitilir. Epoch sayısı, batch boyutu, resim boyutu gibi hiperparametreler ayarlanır.
4.  **Doğrulama ve Test:** Eğitim sonrası en iyi model (`best.pt`) kullanılarak doğrulama ve test setleri üzerinde performans değerlendirmesi yapılır (mAP gibi metrikler hesaplanır).
5.  **Sonuçların Paketlenmesi:** Tüm eğitim çıktıları (ağırlıklar, loglar, tahmin görselleri vb.) bir `.zip` dosyası olarak arşivlenir.

**Not:** Uygulamada kullanılacak `best_trained_model.pt` dosyası bu notebook çalıştırılarak elde edilir.

## Önemli Yapılandırma Noktaları

*   **`app.py` içinde:**
    *   `MODEL_PATH`: Kullanılacak `.pt` model dosyasının yolu.
    *   `CONFIDENCE_THRESHOLD`: Nesne tespiti için minimum güven skoru.
    *   `PROCESS_EVERY_N_FRAMES`: Performans için her kaçıncı karede bir model çıkarımı yapılacağı. Daha yüksek değer, daha az CPU kullanımı ama daha az akıcı tespit anlamına gelir.
*   **`index_phone.html` içinde:**
    *   `UPLOAD_INTERVAL_MS`: Sunucuya ne sıklıkla kare gönderileceği (milisaniye). Daha düşük değer, daha akıcı bir deneyim sunar ancak daha fazla ağ trafiği ve sunucu yükü oluşturur.

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.
