###########################################################################
#Çalışanların deneyim yılı ve maaş bilgileri verilmiştir.
#1-Verilen bias ve weight’e göre doğrusal regresyon model denklemini oluşturunuz.
#Bias = 275, Weight= 90 (y’ = b+wx)
#2-Oluşturduğunuz model denklemine göre tablodaki tüm deneyim yılları için maaş tahmini yapınız.
#3-Modelin başarısını ölçmek için MSE, RMSE, MAE skorlarını hesaplayınız
############################################################################

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df =  pd.read_excel(r".6-machine_learning/RM için Başarı Değerlendirme/maas_tahmin.xlsx")
df

# Verilen bias ve weight değerleri
bias = 275
weight = 90

deneyim_yili = np.array(df["deneyim_yili(x)"])
maas = np.array(df["maas(y)"])

# Doğrusal regresyon modeli oluşturma
maas_tahmini = bias + weight * deneyim_yili

#  Maaş tahminlerini yazdırma
print("| deneyim_yili(x)   | maas(y)     | maas_tahmini(y2)  | ")
for i in range(len(deneyim_yili)):
    print(f"| {deneyim_yili[i]:15}       |     {maas[i]:5}      |   {maas_tahmini[i]:5}  |")

# Modelin başarısını ölçmek için MSE, RMSE, ve MAE hesaplama
hata = maas - maas_tahmini
hata_kareleri = hata ** 2
mutlak_hata = np.abs(hata)

MSE = np.mean(hata_kareleri)
RMSE = np.sqrt(MSE)
MAE = np.mean(mutlak_hata)

print("\nModel Performansı:")
print(f"MSE:    |  {MSE}      |")
print(f"RMSE:   |  {RMSE}     |")
print(f"MAE:    |   {MAE}   |")


# Verilen veriler
veriler = {
    'deneyim_yili': [5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1],
    'maas': [600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380],
    'Tahmini Maaş': [maas_tahmini[i] for i in range(len(deneyim_yili))],
    'Hata': [hata[i] for i in range(len(deneyim_yili))],
    'Hata Kareleri': [hata_kareleri[i] for i in range(len(deneyim_yili))],
    'Mutlak Hata': [mutlak_hata[i] for i in range(len(deneyim_yili))]
}

# Verileri DataFrame'e dönüştürme
df = pd.DataFrame(veriler)

# DataFrame'i Excel dosyasına yazma
df.to_excel('maas_tahmini.xlsx', index=False)
df

import seaborn as sns

sns.scatterplot(x='deneyim_yili', y='maas', data=df)
plt.xlabel('Deneyim Yılı (x)')
plt.ylabel('Maaş (y)')
plt.show()

sns.histplot(data=df['Hata'])
plt.xlabel('Hata')
plt.ylabel('Sıklık')
plt.show()

sns.scatterplot(x='deneyim_yili', y='Hata', data=df)
plt.xlabel('Deneyim Yılı (x)')
plt.ylabel('Hata')
plt.show()


ortalama_maas = df.groupby('deneyim_yili')['maas'].mean()
ortalama_tahmini_maas = df.groupby('deneyim_yili')['Tahmini Maaş'].mean()

plt.plot(ortalama_maas.index, ortalama_maas, label='Ortalama Gerçek Maaş')
plt.plot(ortalama_tahmini_maas.index, ortalama_tahmini_maas, label='Ortalama Tahmini Maaş')
plt.xlabel('Deneyim Yılı (x)')
plt.ylabel('Maaş (y)')
plt.legend()
plt.show()


