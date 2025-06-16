import os
import argparse
from pdf2image import convert_from_path

def convert_first_page_to_jpg(input_folder, output_folder, file_extension=".pdf"):
    """
    Belirtilen klasördeki tüm PDF dosyalarının ilk sayfasını JPG formatına dönüştürür.
    
    Args:
        input_folder (str): Kaynak PDF dosyalarının bulunduğu klasör yolu
        output_folder (str): JPG dosyalarının kaydedileceği klasör yolu
        file_extension (str): İşlenecek dosya uzantısı (varsayılan: ".pdf")
    """
    # Çıktı klasörünü oluştur (eğer yoksa)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Çıktı klasörü oluşturuldu: {output_folder}")

    # Klasördeki tüm dosyaları al
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(file_extension.lower())]
    
    if not files:
        print(f"Belirtilen klasörde '{file_extension}' uzantılı dosya bulunamadı.")
        return
    
    print(f"Toplam {len(files)} adet {file_extension} dosyası bulundu. Dönüştürme başlıyor...")
    
    # Her dosyayı işle
    for i, filename in enumerate(files, 1):
        input_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + ".jpg"
        output_path = os.path.join(output_folder, output_filename)
        
        try:
            # PDF'in ilk sayfasını al
            print(f"[{i}/{len(files)}] Dönüştürülüyor: {filename}")
            images = convert_from_path(input_path, first_page=1, last_page=1)
            
            if images:
                # İlk sayfayı JPG olarak kaydet
                images[0].save(output_path, "JPEG")
                print(f"✓ Kaydedildi: {output_path}")
            else:
                print(f"✗ Hata: {filename} dosyasında sayfa bulunamadı.")
        except Exception as e:
            print(f"✗ Hata: {filename} dosyası dönüştürülemedi. Hata: {str(e)}")

def main():
    # Komut satırı argümanlarını tanımla
    parser = argparse.ArgumentParser(description="PDF dosyalarının ilk sayfasını JPG olarak dönüştür")
    parser.add_argument("-i", "--input", required=True, help="Kaynak dosyaların bulunduğu klasör")
    parser.add_argument("-o", "--output", required=True, help="JPG dosyalarının kaydedileceği klasör")
    parser.add_argument("-e", "--extension", default=".pdf", help="İşlenecek dosya uzantısı (varsayılan: .pdf)")
    
    args = parser.parse_args()
    
    # Fonksiyonu çağır
    convert_first_page_to_jpg(args.input, args.output, args.extension)
    
if __name__ == "__main__":
    main()