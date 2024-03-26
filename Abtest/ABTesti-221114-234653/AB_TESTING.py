'''
#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.
'''



#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.
df = pd.read_excel("V:\.MIUUL\.0 casestudy\.3-measurement_problems\Abtest\ABTesti-221114-234653\Ab_testing.xlsx")
df.head()
df.astype(int)

control_df = pd.read_excel("V:\.MIUUL\.0 casestudy\.3-measurement_problems\Abtest\ABTesti-221114-234653\Ab_testing.xlsx",sheet_name="Control Group")

test_df = pd.read_excel("V:\.MIUUL\.0 casestudy\.3-measurement_problems\Abtest\ABTesti-221114-234653\Ab_testing.xlsx",sheet_name="Test Group")

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.
def check_df(dataframe, head=5):
    print("############## Shape #############")
    print(dataframe.shape)
    print("############## Types #############")
    print(dataframe.dtypes)
    print("############## Tail #############")
    print(dataframe.tail(head))
    print("############## NA #############")
    print(dataframe.isnull().sum())
    print("############## Quantiles #############")
    print(dataframe.describe([0,0.05,0.50,0.95,0.99,1]).T)

check_df(control_df)
check_df(test_df)

control_df.groupby("Click")["Purchase"].mean()
test_df.groupby("Click")["Purchase"].mean()

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.
control_df['Group'] = 'Control'
test_df['Group'] = 'Test'
df = pd.concat([control_df, test_df], ignore_index=True)

'''
#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.
# H0: M1 = M2
# H1: M1 != M2

# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz
'''

df.groupby("Group").agg({"Purchase":"mean"})

'''
#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################
#p-value < 0.05 H0 RED

#p-value > 0.05 H0 REDDEDİLEMEZ

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.
# Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını
# Purchase değişkeni üzerinden ayrı ayrı test ediniz




# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz


# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.
'''

# Normallik Varsayımı
test_stat, pvalue = shapiro(df.loc[df["Group"] == "Control", "Purchase"])
print('Test Stat : %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Group"] == "Test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
    #Test Stat : 0.9773, p-value = 0.5891
    #Test Stat = 0.9589, p-value = 0.1541

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# Varyans Homojenliği
test_stat, pvalue = levene(df.loc[df["Group"] == "Control", "Purchase"],  ## levene iki farklı grubu gönder ben sana bu iki grubun
                           df.loc[df["Group"] == "Test", "Purchase"])              # varyans homojenliği sağlayıp sağlayamadığını ifade edeyim.
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
    #Test Stat = 2.6393, p-value = 0.1083

# Normallik Varsayımı ve Varsayım Homojenliği sağlandığı için t (bağımsız iki örneklem) testi uygulanacaktır.
test_stat, pvalue = ttest_ind(df.loc[df['Group'] == 'Control', 'Purchase'],
                              df.loc[df['Group'] == 'Test', 'Purchase'],
                              equal_var=True)
print('Test stat : %.4f, P-value : %.4f' % (test_stat, pvalue))
    #Test stat : -0.9416, P-value : 0.3493

#p-value < ise 0.05'ten HO RED.
#p-value < değilse 0.05 H0 REDDEDILEMEZ.
#H0 Reddedilemez. Maximum bidding ve Average bidding arasında fark yoktur.

##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

#Bağımsız iki örneklem t testi.Normallik ve varyans homojenliği varsayımları sağlandığı için.
#Normallik varsayımı ve varyans homojenliği varsayımları sağlandığı için bu testi kullanmak uygun oldu.


# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

# Yaptığımız test sonuçlarına göre, Maximum Bidding ve Average Bidding arasında istatistiksel olarak anlamlı bir fark olmadığı için,
# bombabomba.com'un hangi stratejiyi kullanmaya devam edeceğine dair bir seçim yaparken diğer faktörleri (örneğin, reklam bütçesi, hedef kitle vb.)
# göz önünde bulundurması önerilir.
# Dolayısıyla, müşteriye bu iki teklif türü arasında dönüşümü artırmak açısından bir tercih yapmalarını önermekte zorlanırız. Ancak,
# işletmenin ihtiyaçlarına, hedeflerine ve hedef kitlesine bağlı olarak, her iki teklif türünün avantajlarını ve dezavantajlarını
# değerlendirerek karar vermeleri daha uygun olabilir.