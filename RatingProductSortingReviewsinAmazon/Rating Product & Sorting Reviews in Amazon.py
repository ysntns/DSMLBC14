
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı



###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################
# Veri setini oku.
df = pd.read_csv("V:\.MIUUL\.0 casestudy\.3-measurement_problems\RatingProductSortingReviewsinAmazon\Amazon_review.csv")

# Ürünün ortalama puanını hesapla.
average_rating = df["overall"].mean()

df["overall"].mean()



###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################
df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = df["reviewTime"].max()

df["recency_rating_review"] = (current_date - df["reviewTime"]).dt.days

df["recency_cut"]= pd.qcut(df["recency_rating_review"], 4, labels= ["q1", "q2", "q3", "q4" ])

def time_based_weighted_average(dataframe, w1=30, w2=28, w3=26, w4=16):
    return dataframe.loc[df["recency_cut"] == "q1", "overall"].mean() * w1 / 100 + \
           dataframe.loc[df["recency_cut"] == "q2", "overall"].mean() * w2 / 100 + \
           dataframe.loc[df["recency_cut"] == "q3", "overall"].mean() * w3 / 100 + \
           dataframe.loc[df["recency_cut"] == "q4", "overall"].mean() * w4 / 100

time_based_weighted_average(df)



###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################

###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()

###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################
def score_pos_neg_diff(pos, neg):
    return pos - neg

df["score_pos_neg_diff"] = score_pos_neg_diff(df["helpful_yes"], df["helpful_no"])


def score_average_rating(pos, neg):
    if pos + neg == 0:
        return 0
    return pos / (pos + neg)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)



def wilson_lower_bound(pos, neg, confidence=0.95):
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)



df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.head()

##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################
df.sort_values("wilson_lower_bound", ascending=False).head(20)

# WLB ile yapılan sıralamada ilk dörtte en çok oy alan yorumlar yer alıyor bu da bize WLB'un social proofu dikkate
# aldığını gösterir. Aynı zamanda faydalı bulunma oranı da dikkate alınmıştır. Aynı zamanda score_average_ratinge göre
# sıralama yapsaydık gözlem 5 gözlem 4 ün önüne geçecekti fakat burda 4. gözlemde daha çok oylama yapıldığı için WBL hesabında 5'in önüne geçmiştir.