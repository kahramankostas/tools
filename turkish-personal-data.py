import random
from datetime import datetime, timedelta
import string

def generate_tc_kimlik():
    return str(random.randint(10000000000, 99999999999))

def generate_phone():
    return f"0{random.randint(500,599)}{random.randint(1000000,9999999)}"

def generate_landline():
    return f"0212{random.randint(1000000,9999999)}"

def random_date(start_year=1960, end_year=2000):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randrange(delta.days)
    return (start + timedelta(days=random_days)).strftime("%d.%m.%Y")

# Turkish name lists
turkish_first_names = ["Ahmet", "Mehmet", "Ali", "Ayşe", "Fatma", "Emine", "Mustafa", "İbrahim", "Zeynep", "Hatice"]
turkish_last_names = ["Yılmaz", "Kaya", "Demir", "Şahin", "Çelik", "Yıldız", "Özdemir", "Arslan", "Doğan", "Kılıç"]
turkish_cities = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana", "Konya", "Gaziantep", "Şanlıurfa", "Mersin"]

# Skills and qualifications
computer_programs = ["MS Office", "AutoCAD", "Python", "Java", "Adobe Photoshop", "SPSS", "SAP", "Oracle"]
instruments = ["Bağlama", "Ud", "Kanun", "Ney", "Gitar", "Keman", "Piyano"]
folk_dances = ["Horon", "Zeybek", "Halay", "Çiftetelli", "Karşılama", "Bar"]
projects = ["AB Projesi", "TÜBİTAK Projesi", "Sosyal Sorumluluk Projesi", "Ar-Ge Projesi"]
publications = ["Akademik Makale", "Konferans Bildirisi", "Kitap Bölümü", "Teknik Rapor"]

def generate_records(num_records):
    records = []
    header = "TC Kimlik No,Adı,Soyadı,Baba Adı,Ana Adı,Doğum Yeri,Doğum Tarihi,Cinsiyeti,Medeni Durumu,Çocuk Sayısı,Başvurulan Ülke/Ülke Grubu,Ev Telefonu,İş Telefonu,Cep Telefonu,E-posta,Sürücü Belgesi,Halk Oyunları,Enstrüman,Bilgisayar Programları,Projeler,Yayınlar,Diğer Yetenekler"
    records.append(header)

    for _ in range(num_records):
        first_name = random.choice(turkish_first_names)
        last_name = random.choice(turkish_last_names)
        gender = "E" if first_name in ["Ahmet", "Mehmet", "Ali", "Mustafa", "İbrahim"] else "K"
        
        record = [
            generate_tc_kimlik(),
            first_name,
            last_name,
            random.choice(turkish_first_names),  # Baba Adı
            random.choice(turkish_first_names),  # Ana Adı
            random.choice(turkish_cities),
            random_date(),
            gender,
            random.choice(["Evli", "Bekar", "Boşanmış"]),
            str(random.randint(0, 4)),
            random.choice(["AB", "Amerika", "Uzak Doğu", "Orta Doğu"]),
            generate_landline(),
            generate_landline(),
            generate_phone(),
            f"{first_name.lower()}.{last_name.lower()}@email.com",
            random.choice(["B", "A", "BE", "Yok"]),
            random.choice(folk_dances),
            random.choice(instruments),
            ", ".join(random.sample(computer_programs, k=random.randint(1, 3))),
            ", ".join(random.sample(projects, k=random.randint(0, 2))),
            ", ".join(random.sample(publications, k=random.randint(0, 2))),
            random.choice(["Yabancı Dil", "Spor", "Fotoğrafçılık", ""])
        ]
        records.append(",".join(record))
    
    return "\n".join(records)

# Get number of records from user
try:
    num_records = int(input("Kaç adet kayıt oluşturmak istiyorsunuz? "))
    if num_records <= 0:
        print("Lütfen pozitif bir sayı giriniz.")
    else:
        # Generate and save records to file
        data = generate_records(num_records)
        with open('turkish_personal_data.csv', 'w', encoding='utf-8') as f:
            f.write(data)
        print(f"{num_records} adet kayıt başarıyla oluşturuldu ve 'turkish_personal_data.csv' dosyasına kaydedildi.")
except ValueError:
    print("Lütfen geçerli bir sayı giriniz.")
