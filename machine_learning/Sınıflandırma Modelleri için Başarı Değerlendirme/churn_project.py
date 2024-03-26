############################################
# Görev 1
############################################

############################################
# Müşterinin churn olup olmama durumunu tahminleyen bir
# sınıflandırma modeli oluşturulmuştur. 10 test verisi gözleminin
# gerçek değerleri ve modelin tahmin ettiği olasılık değerleri
# verilmiştir.
############################################
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel(r".6-machine_learning/Sınıflandırma Modelleri için Başarı Değerlendirme/churn_project.xlsx")
df

# Eşik değerini 0.5 alarak confusion matrix oluşturma
df['Model Tahmini'] = df['Model Olasılık Tahmini (1 sınıfına ait olma olasılığı)'].apply(lambda x: 1 if x >= 0.5 else 0)

# Confusion Matrix oluşturma
confusion_matrix = pd.crosstab(df['Gerçek Değer'], df['Model Tahmini'], rownames=['Gerçek Değer'], colnames=['Model Tahmini'])

print("Confusion Matrix:")
print(confusion_matrix)

# Doğruluk (Accuracy), Duyarlılık (Recall), Kesinlik (Precision) ve F1 Skorlarını hesaplama
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

accuracy = accuracy_score(df['Gerçek Değer'], df['Model Tahmini'])
recall = recall_score(df['Gerçek Değer'], df['Model Tahmini'])
precision = precision_score(df['Gerçek Değer'], df['Model Tahmini'])
f1 = f1_score(df['Gerçek Değer'], df['Model Tahmini'])

print("\nDoğruluk (Accuracy):", accuracy)
print("Duyarlılık (Recall):", recall)
print("Kesinlik (Precision):", precision)
print("F1 Skoru:", f1)


# Sonuçları df DataFrame'ine sütun olarak ekleme
df['Doğruluk (Accuracy)'] = accuracy
df['Duyarlılık (Recall)'] = recall
df['Kesinlik (Precision)'] = precision
df['F1 Skoru'] = f1

# Sonuçları churn_project.xlsx dosyasına kaydetme
df.to_excel("churn_project_with_results.xlsx", index=False)

############################################################

df2 = pd.read_excel(r"churn_project_with_results.xlsx")
df2

############################################
# Görev 2
############################################

############################################
#Banka üzerinden yapılan işlemler sırasında dolandırıcılık işlemlerinin yakalanması amacıyla sınıflandırma modeli oluşturulmuştur.
# %90.5 doğruluk oranı elde edilen modelin başarısı yeterli bulunup model canlıya alınmıştır.
# Ancak canlıya alındıktan sonra modelin çıktıları beklendiği gibi olmamış, iş birimi modelin başarısız olduğunu iletmiştir.
# Aşağıda modelin tahmin sonuçlarının karmaşıklık matriksi verilmiştir. Buna göre;
############################################

# Accuracy, Recall, Precision, F1 Skorlarını hesaplayınız.
#Veri Bilimi ekibinin gözden kaçırdığı durum ne olabilir yorumlayınız.
'''

                        | Fraud (1) | Non-Fraud (0) |
|---------------|-----------|---------------|
| Fraud (1)        |     5     |      5                 |       10      |
| Non-Fraud (0) |     90    |     900               |       990    |
                      |     95    |     905               |

Bu matristen yola çıkarak aşağıdaki ölçümleri hesaplama

Doğruluk (Accuracy)
Duyarlılık (Recall) (Fraud sınıfı için)
Kesinlik (Precision) (Fraud sınıfı için)
F1 Skoru (Fraud sınıfı için)
'''


# Accuracy hesaplama
accuracy = 905 / 1000
print("Accuracy =", accuracy)

# Duyarlılık (Recall) hesaplama
recall = 0.5
print("Recall (Duyarlılık) =", recall)

# Kesinlik (Precision) hesaplama
precision = 5 / 95
print("Precision (Kesinlik) =", precision)

# F1 Skoru hesaplama
f1_score = 2 * (precision * recall) / (precision + recall)
print("F1 Skoru =", f1_score)


'''
Veri Bilimi ekibinin gözden kaçırdığı durum muhtemelen modelin dengesiz sınıf dağılımıyla başa çıkma yeteneği olabilir. 
Çünkü verilen matriste, Non-Fraud (0) sınıfının Fraud (1) sınıfına göre çok daha fazla olduğu görülmektedir. 
Bu durumda model, genellikle Non-Fraud (0) sınıfını tahmin etme eğiliminde olabilir ve dolayısıyla
 Fraud (1) sınıfını doğru bir şekilde tahmin etme başarısı düşük olabilir. Bu tür dengesizlikler, 
 model performansını etkileyebilir ve modelin canlıya alındıktan sonra beklenen başarıyı gösterememesine neden olabilir. 
 Bu tür durumlarla başa çıkmak için dengesiz veri kümesi tekniklerini kullanarak modeli eğitmek veya daha fazla 
 Fraud (1) örneği toplamak gibi çözümler değerlendirilebilir.
'''



# Yeni sonuçları içeren DataFrame'i oluşturma
yeni_veriler = pd.DataFrame({
    'Metrik': ['Accuracy', 'Recall', 'Precision', 'F1 Score'],
    'Deger': [accuracy, recall, precision, f1_score]
})

# Yeni verileri önceki verilerle birleştirme
birlesik_veri = pd.concat([df2, yeni_veriler], ignore_index=True)

# Birleştirilmiş verileri Excel dosyasına kaydet
birlesik_veri.to_excel("churn_project_with_result.xlsx", index=False)


import seaborn as sns

sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.show()

sns.displot(data=df, x='Model Olasılık Tahmini (1 sınıfına ait olma olasılığı)', hue='Gerçek Değer', kind='kde')
plt.show()

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(df['Gerçek Değer'], df['Model Olasılık Tahmini (1 sınıfına ait olma olasılığı)'])

plt.plot(recall, precision, label='Precision-Recall Eğrisi')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

df['Churn Tahmini'] = df['Model Olasılık Tahmini (1 sınıfına ait olma olasılığı)'].apply(lambda x: 1 if x >= 0.5 else 0)

churn_tahmini_gruplar = df.groupby('Churn Tahmini')

churn_oranlari = churn_tahmini_gruplar['Gerçek Değer'].mean()

lift = churn_oranlari / churn_oranlari.iloc[0]

plt.plot(lift.index, lift.values)
plt.xlabel('Churn Tahmini')
plt.ylabel('Lift')
plt.show()
