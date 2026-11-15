# Hafta 13 / Demo Day — Promoter-GPT

Hafta 7-8'de sıfırdan yazdığım karakter seviyesi GPT'yi, Türkçe metin yerine
insan promoter dizilerine uyguladım. Amaç tek bir soruyu ölçmek: **yalnızca
sonraki bazı tahmin etmeyi öğrenen bir model, bir dizinin promoter olup
olmadığını ayırt edebilir mi?** Eğitimde hiçbir etiket kullanılmıyor.

## Veri

Genomic Benchmarks `human_nontata_promoters` setini kullandım: toplam
36.131 dizi, her biri 251 bp, hepsi tam uzunlukta —
farklı uzunlukta tek bir dizi yok. Alfabe A, C, G, T ve tüm veri setinde tek
bir N; BOS token'ı ile birlikte vocab 6.

| bölüm | pozitif (promoter) | negatif |
| --- | ---: | ---: |
| eğitim | 14.742 | 12.355 |
| test | 4.915 | 4.119 |

Model **yalnızca eğitim bölümündeki pozitif dizilerde** eğitildi
(3,70 Mbase). Negatifler eğitimde hiç görülmedi; sadece
değerlendirmede kullanıldı. Sınıflandırmanın "etiketsiz" olması buradan
geliyor.

Pozitif ve negatif dizilerin GC oranı belirgin şekilde farklı (eğitim
bölümünde 0,6267 ve 0,4803). Bu farkı eğitimden **önce** not
ettim, çünkü sadece-GC baseline'ının güçlü olacağını baştan biliyordum; asıl
soru modelin bunun ötesine geçip geçmediği.

## Kurulum

Mimari hafta 8'deki `v2.py` ile aynı: token ve öğrenilen pozisyon embedding'i,
6 adet pre-LN transformer bloğu (n_embd 384,
6 head, dropout 0,20), final LayerNorm,
lm_head. Toplam 10,74M parametre.

Diziler bağımsız kayıtlar olduğu için uç uca eklenmedi: her dizinin başına BOS
konup kendi örneği olarak batch'lendi. Böylece model olmayan dizi sınırları
arası geçiş öğrenmiyor ve dizi başına loss tanımlı hale geliyor. `block_size`
251, yani bir promoter bağlama tam sığıyor.

Hafta 8'den iki sapma var. Birincisi dikkat, el yazımı tril maskesi yerine
`scaled_dot_product_attention` ile hesaplanıyor; matematiği aynı, füzyonlu
çekirdek MPS'te daha hızlı. İkincisi bir düzeltme: hafta 8'de ölçeklendirme
`n_embd ** -0.5` idi, oysa standart formülasyon `head_size ** -0.5` ister. Multi head attention'a geçerken oluşan bir kayma; 6 head attention logit'lerini
1/√6 kat fazla küçültüyordu, yani dikkat dağılımı olması gerekenden
dah smooth. Buradaki iki yol da doğru ölçeği kullanıyor.

## Eğitim

3.268 optimizer step, 52,50M token
(≈ 14,2 epoch), eğitim adımlarının toplam süresi
35,0 dakika, ortalama 24.991 token/saniye.
Değerlendirme ve üretim süresi bu hıza dahil değil.

İlk ölçüm 1,8015 nat, yani
2,5991 bits/base. Bu tesadüf değil:
altı token üzerinde düzgün dağılımın entropisi ln(6) = 1,7918
nat. Model başlangıçta BOS ve N'nin hiç geçmediğini de bilmiyor, o yüzden
2,0'ın üstünde başlıyor.

Dört referans seviyesi var ve hangisine karşı ölçtüğünüz sonucu değiştiriyor:

| referans | bits/base |
| --- | ---: |
| başlangıç — 6 token üzerinde düzgün | 2,585 |
| rastgele — dört baz üzerinde düzgün | 2,000 |
| unigram — yalnızca baz frekansları, bağlam yok | 1,953 |
| **model — en iyi doğrulama** | **1,835** |

Unigram satırı önemli. Promoter'lar GC-zengin olduğu için
(A 0,182, C 0,303, G 0,323, T 0,191)
hiçbir bağlam kullanmayan, yalnızca baz frekanslarını bilen bir model dahi
2,0'ın altına, 1,953 bits/base'e iner. Yani
"rastgeleyi geçtim" demek fazla cömert bir ifade olurdu.

Model rastgeleyi 0,165
bits/base geçiyor, ama bunun yalnızca
0,047 kadarı baz
frekanslarından gelen bedava kazanç. Bağlama bakarak kazandığı kısım
**0,118 bits/base**,
yani toplam kazancın
%71'i.
Modelin gerçekten öğrendiği şey bu.

Doğrulama loss'u koşunun ikinci yarısında neredeyse hiç iyileşmedi: ilk yarının en iyisi 1,2745 nat, ikinci yarının en iyisi 1,2719 nat — yalnızca %0,21 kazanç. Eğitim ve doğrulama arasındaki fark da küçük (0,0130 nat), yani aşırı öğrenme yok; model bu kapasite ve bu veriyle yapabileceğine ulaşmış, düzleşmiş. Daha uzun eğitmek anlamlı bir fark yaratmazdı; iyileştirme için ya daha fazla veri ya da mimari değişikliği gerekiyor. Aşağıdaki tüm değerlendirmeler en iyi doğrulama checkpoint'inden (2.800. adım) hesaplandı, son adımdan değil.

Loss eğrisi: `results/loss.png`.

## Sonuç 1 — etiketsiz sınıflandırma

Test bölümündeki her dizi için baz başına ortalama loss hesaplandı. Düşük loss
= modele "tanıdık" = promoter. Diziler bu skora göre sıralanıp AUROC
hesaplandı. AUROC, beraberliklere duyarlı Mann-Whitney U ile sıfırdan yazıldı.

| yöntem | AUROC |
| --- | ---: |
| model (dizi başına loss) | **0,7263** |
| sadece GC oranı | 0,7959 |

Pozitif dizilerin ortalaması 1,8423 bits/base,
negatiflerin 1,9170 bits/base. Dağılımlar:
`results/auroc_hist.png`.

Sadece-GC baseline'ı modeli **0,0696 AUROC** puanı geçiyor. Yani bu eğitim bütçesinde model, tek bir özet istatistiğin verdiği bilgiyi dahi tam olarak kullanamamış. Bu bir başarısızlık değil, ölçümün kendisi: küçük veri ve kısa eğitimle generative bir modelin ayırt edici gücü basit bir sayımla yarışamıyor.

## Sonuç 2 — in-silico saturation mutagenesis

En düşük loss'lu test pozitifini alıp her pozisyonda her alternatif bazı tek
tek denedim (753 mutant), her biri için dizinin
log-likelihood'undaki değişimi ölçtüm. Bu, büyük DNA dil modellerinin varyant
etkisi tahmini için kullandığı yöntemin aynısı: mutasyon sonrası olasılık
değişimi. Maskeleme değil — maskeleme modelin MLM ile eğitilmiş olmasını
gerektirir, bu model otoregresif.

Orijinal dizi 1,2498 bits/base, toplam
Δ NLL ortalaması 1,9789 nat.

**Baz tercihi net ve GC yönünde:**

| mutasyon hedefi | ortalama Δ NLL (nat) |
| --- | ---: |
| T | 3,1221 |
| A | 2,2952 |
| C | 1,2494 |
| G | -0,3139 |

Herhangi bir bazı T veya A'ya çevirmek diziyi belirgin biçimde daha az olası
yapıyor; G'ye çevirmek ise hafifçe daha olası yapıyor. Isı haritasında bu,
A ve T satırlarının baştan sona kırmızı, G satırının nötr olması olarak
görünüyor.

**Varyans ayrışımı bu koşuda hesaplanmadı.** `analyze.py` yeniden koşturulduğunda baz kimliği ile pozisyonun payları buraya gelecek.

## Modelin skoru GC'den ibaret mi? Hayır

Akla gelen ilk açıklama, modelin skorunun örtük bir GC ölçümü olduğuydu.
Bunu doğrudan test ettim: dizi başına loss ile GC oranı arasındaki korelasyon
-0,608, yani
r² = 0,370. GC, skordaki varyansın
yalnızca %37'sini açıklıyor.
GC'yi doğrusal olarak çıkardıktan sonra kalan artığın AUROC'u
0,5600 — zayıf, ama 0,5 değil.

Yani hipotez çürüdü: **skor GC'den ibaret değil, GC'den daha kötü.** Model
kompozisyonun ötesinde bir şey ölçüyor, ama ölçtüğü şey bu ayrım için GC
oranından daha zayıf bir özellik.

Pozitif ve negatiflerde korelasyonun ayrılması da bilgi veriyor: pozitiflerde
-0,696, negatiflerde
-0,117. Model yalnızca pozitiflerde
eğitildiği için orada skoru "bu dizi eğitim dağılımına ne kadar tipik"
sorusunu ölçüyor ve bu GC ile güçlü biçimde ilişkili. Negatiflerde ise model
dağılım dışında kalıyor ve skorunu başka bir şey yönetiyor.

Tek modelli kurulumun yapısal zayıflığı burada: dizi başına ortalama loss,
`log P(dizi | promoter)` tahminidir, ama sınıflandırma için gereken
`log P(dizi | promoter) − log P(dizi | negatif)`. İkinci terim olmadığı için
her iki sınıfta ortak olan ne varsa skorda kalıyor.

Doğru düzeltme olabilirlik oranı: negatiflerde ikinci bir model eğitip iki
modelin log-olasılık farkıyla skorlamak. O zaman ortak terim büyük ölçüde
sadeleşir. Bugünkü bütçeye sığmadı, sırada o var.

## Teşhisten çıkan düzeltme

Eksik olan terim belliydi: model `log P(dizi | promoter)` hesaplıyor, oysa
sınıflandırma için gereken `log P(dizi | promoter) − log P(dizi | negatif)`.
Negatif sınıfta ikinci bir GPT eğitip (20 dakika) skoru iki modelin farkı
yaptım. AT-zengin bir dizi **her iki modele de** kolay geldiği için kemeri
yaratan ortak terim farkta sadeleşiyor.

| yöntem | katman | AUROC |
| --- | :---: | ---: |
| model loss'u — dizi başına ortalama | 1 | 0,7263 |
| GC oranı | 1 | 0,7959 |
| olabilirlik oranı — iki modelin farkı | 2 | **0,9208** |

Skorun GC ile korelasyonu -0,608 iken 0,570'ye döndü ve kemer katsayısı 0,367 oldu: skor artık GC ile aynı yönde ve monoton. Tahmin edilen mekanizma, tahmin edildiği gibi davrandı.

**Katmanlar aynı tabloda sıralanamaz.** Katman 1 hiçbir yerde etiket
kullanmıyor. Katman 2 etiketleri ikinci modeli *kurmak* için kullanıyor, ama
skorlarken kullanmıyor. Katman 3 doğrudan etiketle eğitilen probe'lar. Farklı
soruların cevapları; aynı yarışın sonuçları değil.

## Sınırlar

**Causal model pozisyon eğilimi yaratıyor.** Model yalnızca sola bakıyor, yani
`i` pozisyonundaki bir mutasyon sadece `i` ve sonrasındaki tahminleri
değiştirebilir. Dizinin başındaki bir mutasyonun etkileyeceği
251 pozisyon varken, sonundakinin birkaç pozisyonu var.
Ölçülen korelasyon 0,056 (|Δ| ile aşağı
akış pozisyon sayısı arasında). Isı haritasında hem ham hem aşağı akış
sayısına normalize edilmiş hali var. Çift yönlü modeller bu sorunu maskeleme
ile çözüyor; otoregresif modeller varyantı dizinin sonuna yerleştiriyor.

**GC karışması.** Promoter'lar GC-zengin olduğu için ayırt edici gücün bir
kısmı kompozisyondan geliyor. Sadece-GC baseline'ı tam bu yüzden hesaplandı,
ama bu karşılaştırma farkın her bir nedene düşen payını ayırmıyor.

**Ölçek.** 10,74M parametre ve 3,70 Mbase
veri. Alanın modelleri milyarlarca parametre ve genom ölçeğinde veri
kullanıyor. Buradaki sonuçlar mekanizmayı gösteriyor, rekabetçi bir performans
iddiası taşımıyor.

**Tek eğitim koşusu.** Tohum değiştirilmedi, hiperparametre taraması
yapılmadı. Raporlanan sayıların belirsizliği ölçülmedi.

## Sırada ne var

- **Olabilirlik oranı**: negatiflerde ikinci bir model eğitip iki modelin
  log-olasılık farkıyla skorlamak. Yukarıdaki analize göre tek modelli
  kurulumun asıl eksiği bu
- Veri verimliliği eğrisi: verinin %10, %30 ve %100'ü ile eğitip AUROC'u
  veriye karşı çizmek
- Tokenizer ablasyonu: tek baz yerine k-mer vocab, hafta 9'da yazdığım
  tokenizer ile
- Ters tümleyen tutarlılığı: model bir diziye ve onun ters tümleyenine benzer
  olasılık veriyor mu? Mimaride böyle bir indüktif önyargı yok
- Aşırı öğrenmeyi azaltmak için daha küçük model

## Çalıştırma

```bash
python3 prepare.py
python3 -u train_dna.py --minutes 35
python3 -u analyze.py
python3 make_report.py
```

Veri, checkpoint'ler ve grafikler git'e dahil değil.
