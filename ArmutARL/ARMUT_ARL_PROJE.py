
#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################
# Veri Seti
#########################
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih


# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
from setuptools import setup

setup()

############################################
# Görev 1: Veriyi Hazırlama
############################################
#Adım 1: Online Retail II veri setinden 2010-2011 sheet’ini okutunuz.
df_ = pd.read_csv(".4-recommendation _Systems/ArmutARL/armut_data.csv")
df = df_.copy()
df.head()
df.describe().T
df.isnull().sum()
df.shape

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.
df['Hizmet'] = df['CategoryId'].astype(str) + "_" + df['ServiceId'].astype(str)

# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_"
# ile birleştirirek ID adında yeni bir değişkene atayınız.
df['New_Date'] = pd.to_datetime(df['CreateDate']).dt.to_period('M')
df.head()

# UserID ve yeni oluşturduğumuz tarih bilgisini "_" ile birleştirerek bir sepet ID'si oluşturalım
df['SepetID'] = df['UserId'].astype(str) + "_" + df['New_Date'].astype(str)
df.head()

df["SepetID"] = df["SepetID"].astype(str)

#########################
# GÖREV 2: Birliktelik Kuralları Üretiniz
#########################
# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.
sepet_df =df.pivot_table(index='SepetID', columns='Hizmet', aggfunc='size', fill_value=0)
sepet_df.head()


# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

# Adım 2: Birliktelik kurallarını oluşturunuz.
df_armt = df[df["Hizmet"].isin(["değer1", "değer2", "değer3"])]

df_armt.groupby(['SepetID', 'Hizmet']).agg({"CategoryId": "sum"}).head(20)

df_armt.groupby(['SepetID', 'Hizmet']).agg({"CategoryId": "sum"}).unstack().iloc[0:5, 0:5] #buraları değişken isimlerine çevir.

df_armt.groupby(['SepetID', 'Hizmet']).agg({"CategoryId": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

df_armt.groupby(['SepetID', 'ServiceId']). \
    agg({"CategoryId": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:10, 0:10]

#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.
def arl_recommender(rules_df, last_service):
    # En son alınan hizmetin önerilerini bulalım
    recommendation = rules_df[rules_df['antecedents'] == frozenset({last_service})]. \
        sort_values(by='lift', ascending=False)
    return recommendation

recommendations = arl_recommender(rules, '2_0')
recommendations




def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[dataframe["ServiceId"] > 0]
    dataframe = dataframe[dataframe["CategoryId"] > 0]
    replace_with_thresholds(dataframe, "ServiceId")
    replace_with_thresholds(dataframe, "CategoryId")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['ServiceId', "CategoryId"])['UserId'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['ServiceId', 'CreateDate'])['UserId'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    sepet_df = dataframe[dataframe["Hizmet"] == stock_code][["CategoryId"]].values[0].tolist()
    print(sepet_df)


def create_rules(dataframe, id=True):
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

arl_recommender(rules, 22948, 1)
arl_recommender(rules, 20676, 2)
arl_recommender(rules, 20676, 3)