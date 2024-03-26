
###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500) # yanyana göster sütunları
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# 1. flo_data_20K.csv verisini okuyunuz.
df_ = pd.read_csv("V:\.MIUUL\.0 casestudy\.2-crm_analitica\Flo\.1-flo_rfm\FLOMusteriSegmentasyonu\lodata20k.csv")
df = df_.copy()

# a. İlk 10 gözlem
df.head(10)

# b. Değişken isimleri
df.columns

# e. Değişken tipleri, incelemesi yapınız.
df.dtypes

#boyut bilgisi
df.shape

#Boş değer
df.isnull().sum()

#Betimsel istatistik
df.describe().T

# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

# Müşteri bazında toplam alışveriş sayısı ve harcamayı içeren yeni değişkenleri oluşturma
df['total_shopping_number'] = df['order_num_total_ever_online'] + df["order_num_total_ever_offline"]
df['total_spending'] = df['customer_value_total_ever_online'] + df["customer_value_total_ever_offline"]

# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.dtypes
# Tarih ifade eden değişkenlerin türünü tarihe dönüştür
date_columns = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
# Tarih ifade eden değişkenlerin tipini date'e çevirme
for column in date_columns:
    df[column] = pd.to_datetime(df[column])

df.dtypes



# 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
df.groupby("order_channel").agg({"master_id": "count",
                                 "total_shopping_number": "sum",
                                 "total_spending":"sum"})

# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df["total_spending"].sort_values(ascending=False).head(10)

# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df["total_shopping_number"].sort_values(ascending=False).head(10)

# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
def pre_preparation(df_):
    df_["total_shopping_number"] = df_["order_num_total_ever_online"] + df_["order_num_total_ever_offline"]
    df_["total_spending"] = df_["customer_value_total_ever_online"] + df_["customer_value_total_ever_offline"]

    for col in df_.columns:
        if "date" in col:
            df_[col] = pd.to_datetime(df_[col])

    return df_

prep_df = pre_preparation(df)

# GÖREV 2: RFM Metriklerinin Hesaplanması

# 1. Recency, Frequency, Menoetary tanımlarını yapınız
# Recency = analizin yapıldığı tarih - müşterinin son satın alım yaptığı tarih
# Frequency = satın alım sayısı
# Monetary = müşterinin bıraktığı toplam parasal değer
df["last_order_date"].max()  # 2021-05-30
analysis_date = dt.datetime(2021, 6, 1)
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[ns]')
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

# GÖREV 3: RF ve RFM Skorlarının Hesaplanması
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))

# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm.head()

# GÖREV 5: Aksiyon zamanı!
           # 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

           # 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.
                   # a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
                   # tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers),
                   # ortalama 250 TL üzeri ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına
                   # yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.
target_segments_customer_ids = rfm[rfm["segment"].isin(["champions","loyal_customers"])]["customer_id"]

cust_ids = df[(df["master_id"].isin(target_segments_customer_ids))&
              (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]

cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)

cust_ids.count()
# 2497
cust_ids.shape
# (2497,)

rfm.columns
df.columns
                   # b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
                   # alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
                   # olarak kaydediniz.
target_segments_cust_ids = rfm[rfm["segment"].isin(["cant_loose","hibernating","new_customers"])]["customer_id"]

cust_ids_ = df[(df["master_id"].isin(target_segments_cust_ids)) &
              ((df["interested_in_categories_12"].str.contains("ERKEK"))|(df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]

cust_ids_.count()
# 2771

cust_ids_.to_csv("indirim_hedef_müşteri_ids_.csv", index=False)

###############################################################
# BONUS - Sürecin fonksiyonlaştırılması
###############################################################

# GÖREV 6: Tüm süreci fonksiyonlaştırınız.

def create_rfm(df_):

    # Veriyi önhazırlama
        # Toplam sipariş sayısını ve toplam harcalama değişkenlerini hesaplama
    df_["order_num_total"] = df_["order_num_total_ever_online"] + df_["order_num_total_ever_offline"]
    df_["customer_total_value"] = df_["customer_value_total_ever_online"] + df_["customer_value_total_ever_offline"]

        # Tarih belirlen girdilerin tipini date'e çevirme
    for col in df_.columns:
        if "date" in col:
            df_[col] = pd.to_datetime(df_[col])

    # RFM metriklerini hesaplama
    today_date = dt.datetime(2021, 6, 1)
    df_["Recency"] = [(today_date - date).days for date in df_["last_order_date"]]
    df_["Frequency"] = df_["order_num_total"]
    df_["Monetary"] = df_["customer_total_value"]

        # Hazırladğımız metriklerin rfm isimli bir dataframe'e aktarılması
    rfm = df_[["master_id", "Recency", "Frequency", "Monetary"]]

    # RFM skorlarınının hesaplanması
    rfm["recency_score"] = pd.qcut(x=rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(x=rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(x=rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5])

    # RF skorunun oluşturulması
    rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

    # Segmenlerin RF skoruna göre oluşturulması
    seg_map = {
        r'[1-2][1-2]': 'hibernating',  # birinci ve ikinci elemanında 1 ya da 2 görürsen 'hibernating' diye isimlendir
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',  # birinci ve ikini elemanı 3 ise 'need_attention' diye isimlendir
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

    # indexlerin id'ler olarak ayarlanması
    rfm = rfm.set_index("master_id")

    return rfm


df.head()
rfm_deneme = create_rfm(df)