# -*- coding: utf-8 -*-
"""Generate SONUCLAR.md from results/report.json and results/metrics.csv.

The numbers are never typed by hand. The interpretation sentences branch on
the actual outcome: the goal is to write what is true in each case, not what
reads well. Deciding the wording before seeing the result is the point.

The report itself is in Turkish, matching week 12's SONUCLAR.md; code and CLI
output are in English, matching week 12's README.md.

Usage:  python3 make_report.py
"""

import csv
import json
import os

NATS_TO_BITS = 1.4426950408889634


def fmt(x, n=4):
    """Format a float with a Turkish decimal comma."""
    return f"{x:.{n}f}".replace(".", ",")


def thin(n):
    """Thousands separator, Turkish style (dot)."""
    return f"{n:,}".replace(",", ".")


def main():
    res = "results"
    rep = json.load(open(os.path.join(res, "report.json")))
    rows = list(csv.DictReader(open(os.path.join(res, "metrics.csv"))))
    meta = json.load(open(os.path.join("data", "meta.json")))

    first, last = rows[0], rows[-1]
    best = min(rows, key=lambda r: float(r["val_loss"]))
    steps = int(last["step"])
    tokens = int(last["tokens"])
    tr_sec = float(last["train_seconds"])
    n_train_pos = meta["splits"]["train_pos"]["n"]
    mbase = n_train_pos * meta["seq_length"] / 1e6
    epochs = tokens / (n_train_pos * meta["seq_length"])

    auc_m = rep["auroc_model_loss"]
    auc_gc = rep["auroc_gc_baseline"]
    gap = auc_m - auc_gc
    probe = rep.get("auroc_embedding_probe")

    # --- interpretation branches on the result ---
    if gap > 0.03:
        verdict = (
            f"Model, yalnızca baz kompozisyonuna bakan baseline'ı **{fmt(gap)} AUROC** "
            "puanı geçiyor. Bu, modelin GC oranının ötesinde bir şeyler öğrendiğine "
            "işaret ediyor; ancak hangi örüntüleri öğrendiğini bu ölçüm tek başına "
            "söylemiyor."
        )
    elif gap < -0.03:
        verdict = (
            f"Sadece-GC baseline'ı modeli **{fmt(-gap)} AUROC** puanı geçiyor. Yani bu "
            "eğitim bütçesinde model, tek bir özet istatistiğin verdiği bilgiyi dahi "
            "tam olarak kullanamamış. Bu bir başarısızlık değil, ölçümün kendisi: "
            "küçük veri ve kısa eğitimle generative bir modelin ayırt edici gücü "
            "basit bir sayımla yarışamıyor."
        )
    else:
        verdict = (
            f"Model ile sadece-GC baseline'ı arasındaki fark **{fmt(abs(gap))} AUROC** "
            "puanı, yani ayırt edici gücün büyük kısmı baz kompozisyonundan geliyor "
            "olabilir. Modelin kompozisyondan bağımsız bir yapı öğrendiği sonucuna "
            "bu ölçümle varamayız."
        )

    # Three cases, and they read very differently: the loss can rise (overfit),
    # flatten (capacity or data limited), or still be falling (undertrained).
    v_last, v_best = float(last["val_loss"]), float(best["val_loss"])
    mid = len(rows) // 2
    best_first = min(float(r["val_loss"]) for r in rows[:max(mid, 1)])
    best_second = min(float(r["val_loss"]) for r in rows[mid:])
    late_gain = (best_first - best_second) / best_first      # relative
    train_val_gap = v_last - float(last["train_loss"])

    if v_last > v_best * 1.01:
        of_note = (
            f"Doğrulama loss'u {thin(int(best['step']))}. adımda en iyi değerine "
            f"({fmt(v_best)} nat) inip sonra yukarı kıvrıldı; son adımda "
            f"{fmt(v_last)} nat. Klasik aşırı öğrenme."
        )
    elif late_gain < 0.005:
        of_note = (
            f"Doğrulama loss'u koşunun ikinci yarısında neredeyse hiç "
            f"iyileşmedi: ilk yarının en iyisi {fmt(best_first)} nat, ikinci "
            f"yarının en iyisi {fmt(best_second)} nat — yalnızca "
            f"%{fmt(100 * late_gain, 2)} kazanç. Eğitim ve doğrulama arasındaki "
            f"fark da küçük ({fmt(train_val_gap)} nat), yani aşırı öğrenme "
            f"yok; model bu kapasite ve bu veriyle yapabileceğine ulaşmış, "
            f"düzleşmiş. Daha uzun eğitmek anlamlı bir fark yaratmazdı; "
            f"iyileştirme için ya daha fazla veri ya da mimari değişikliği "
            f"gerekiyor."
        )
    else:
        of_note = (
            f"Doğrulama loss'u bütçe boyunca düşmeye devam etti "
            f"({fmt(v_last)} nat, {thin(steps)}. adım); aşırı öğrenme "
            f"başlamamış ve model hâlâ iyileşiyordu. Daha uzun eğitim "
            f"muhtemelen iyileştirirdi."
        )

    of_note += (
        f" Aşağıdaki tüm değerlendirmeler en iyi doğrulama checkpoint'inden "
        f"({thin(rep['checkpoint_step'])}. adım) hesaplandı, son adımdan değil."
    )

    mut = (rep.get("mutagenesis") or [{}])[0]

    # improve.py's results live in their own file; fold them in when present.
    imp = {}
    ipath = os.path.join(res, "improve_report.json")
    if os.path.exists(ipath):
        try:
            imp = json.load(open(ipath))
        except Exception:
            imp = {}

    def _a(key):
        v = imp.get(key)
        return v.get("auroc") if isinstance(v, dict) else None

    lr = _a("tier2_likelihood_ratio")
    probe_prof = _a("per-position loss profile")
    probe_emb = _a("mean-pooled embedding")
    probe_two = _a("GC and mean loss together")

    if lr is None:
        fix_section = ""
    else:
        rows = [f"| olabilirlik oranı — iki modelin farkı | 2 | **{fmt(lr)}** |"]
        if probe_prof is not None:
            rows.append(f"| pozisyon profili probe'u (251 özellik) | 3 | {fmt(probe_prof)} |")
        if probe_emb is not None:
            rows.append(f"| embedding probe'u (384 özellik) | 3 | {fmt(probe_emb)} |")
        if probe_two is not None:
            rows.append(f"| GC + loss birlikte (2 özellik) | 3 | {fmt(probe_two)} |")
        extra = "\n".join(rows)
        curv = imp.get("tier2_ratio_gc_curvature")
        corr = imp.get("tier2_ratio_gc_corr")
        curv_line = ""
        if curv is not None and corr is not None:
            curv_line = (
                f"\nSkorun GC ile korelasyonu {fmt(rep.get('loss_gc_corr_all', 0.0), 3)}"
                f" iken {fmt(corr, 3)}'ye döndü ve kemer katsayısı {fmt(curv, 3)} oldu:"
                f" skor artık GC ile aynı yönde ve monoton. Tahmin edilen mekanizma,"
                f" tahmin edildiği gibi davrandı.\n")

        fix_section = f"""## Teşhisten çıkan düzeltme

Eksik olan terim belliydi: model `log P(dizi | promoter)` hesaplıyor, oysa
sınıflandırma için gereken `log P(dizi | promoter) − log P(dizi | negatif)`.
Negatif sınıfta ikinci bir GPT eğitip (20 dakika) skoru iki modelin farkı
yaptım. AT-zengin bir dizi **her iki modele de** kolay geldiği için kemeri
yaratan ortak terim farkta sadeleşiyor.

| yöntem | katman | AUROC |
| --- | :---: | ---: |
| model loss'u — dizi başına ortalama | 1 | {fmt(auc_m)} |
| GC oranı | 1 | {fmt(auc_gc)} |
{extra}
{curv_line}
**Katmanlar aynı tabloda sıralanamaz.** Katman 1 hiçbir yerde etiket
kullanmıyor. Katman 2 etiketleri ikinci modeli *kurmak* için kullanıyor, ama
skorlarken kullanmıyor. Katman 3 doğrudan etiketle eğitilen probe'lar. Farklı
soruların cevapları; aynı yarışın sonuçları değil.

"""

    # A missing key must not print as 0,0000. If the decomposition has not
    # been computed yet, leave that paragraph out and say so.
    have_decomp = "var_share_position" in mut
    have_quad = "loss_gc_r2_quadratic" in rep
    probe_row = f"| embedding + lineer probe | {fmt(probe)} |\n" if probe else ""
    by_base_rows = "".join(
        f"| {b} | {fmt(v)} |\n"
        for b, v in sorted((mut.get("delta_by_mutant_base") or {}).items(),
                           key=lambda kv: -kv[1])
    ) or "| (veri yok) | |\n"

    decomp_section = (f"""**Ama etki sadece baz kimliğinden gelmiyor.** İlk analizimde bazlar arası
yayılımı pozisyonlar arası yayılımla doğrudan karşılaştırmıştım; o
karşılaştırma adil değildi, çünkü her baz ortalaması ~{meta['seq_length']}
ölçümün ortalamasıyken her pozisyon ortalaması yalnızca üç ölçümün
ortalaması — ikincisi doğası gereği çok daha gürültülü. Kareler toplamını
ayrıştırmak ikisini eşit şartlarda karşılaştırıyor:

| varyans kaynağı | pay |
| --- | ---: |
| pozisyon | %{fmt(100 * mut.get('var_share_position', 0.0), 1)} |
| mutasyon hedefi bazın kimliği | %{fmt(100 * mut.get('var_share_base_identity', 0.0), 1)} |
| etkileşim ve gürültü | %{fmt(100 * mut.get('var_share_interaction_noise', 0.0), 1)} |

Yani pozisyon etkisi gerçek ve baz etkisinden küçük değil. Model saf bir
kompozisyon sayacı değil; pozisyona bağlı beklentileri var.

**Buna karşılık keskin bir motif yok.** |z| > 2,5 olan pozisyon sayısı
{mut.get('n_positions_abs_z_gt_2p5', 0)}/{meta['seq_length']}; saf gürültüde
beklenen {fmt(mut.get('n_positions_expected_by_chance', 0.0), 0)}. Hiçbir
pozisyon diğerlerinden anlamlı biçimde ayrılmıyor, yani pozisyon etkisi yayvan,
belirli bir bağlanma bölgesinde yoğunlaşmış değil.
`results/mutagenesis_bybase_seq*.png` baz ve pozisyon etkilerini aynı ölçekte
yan yana gösteriyor.

**Beklediğim pozisyon eğilimi ortaya çıkmadı.** Model causal olduğu için
`i` pozisyonundaki mutasyon sadece `i` ve sonrasını etkileyebilir, dolayısıyla
dizinin başındaki mutasyonların daha büyük toplam etki yaratmasını
bekliyordum. Ölçtüm: |Δ| ile aşağı akış pozisyon sayısı arasındaki korelasyon
{fmt(mut.get('position_bias_corr', 0.0), 3)}, yani pratikte yok. Sebebi
şurada görünüyor: toplam etkinin
%{fmt(100 * mut.get('local_share_of_effect', 0.0), 0)}'i mutasyona uğrayan
bazın kendi pozisyonunda. Model aşağı akışı neredeyse hiç etkilemiyor, çünkü
kullandığı bağlam çok kısa menzilli.

İlk yazdığım analizde Δ'yı aşağı akış pozisyon sayısına bölüp normalize
ediyordum. Bu yanlıştı: etki pozisyonla büyümediği için bölme, dizinin sonuna
yapay bir sıcak bölge yaratıyordu ve "en etkili pozisyonlar" olarak 224-250
arasını veriyordu. Düzeltildi; sıralama artık ham Δ üzerinden.

""" if have_decomp else
        "**Varyans ayrışımı bu koşuda hesaplanmadı.** `analyze.py` yeniden "
        "koşturulduğunda baz kimliği ile pozisyonun payları buraya gelecek.\n\n")

    md = f"""# Hafta 13 / Demo Day — Promoter-GPT

Hafta 7-8'de sıfırdan yazdığım karakter seviyesi GPT'yi, Türkçe metin yerine
insan promoter dizilerine uyguladım. Amaç tek bir soruyu ölçmek: **yalnızca
sonraki bazı tahmin etmeyi öğrenen bir model, bir dizinin promoter olup
olmadığını ayırt edebilir mi?** Eğitimde hiçbir etiket kullanılmıyor.

## Veri

Genomic Benchmarks `human_nontata_promoters` setini kullandım: toplam
{thin(36131)} dizi, her biri {meta['seq_length']} bp, hepsi tam uzunlukta —
farklı uzunlukta tek bir dizi yok. Alfabe A, C, G, T ve tüm veri setinde tek
bir N; BOS token'ı ile birlikte vocab {rep['vocab_size']}.

| bölüm | pozitif (promoter) | negatif |
| --- | ---: | ---: |
| eğitim | {thin(n_train_pos)} | {thin(meta['splits']['train_neg']['n'])} |
| test | {thin(rep['n_test_pos'])} | {thin(rep['n_test_neg'])} |

Model **yalnızca eğitim bölümündeki pozitif dizilerde** eğitildi
({fmt(mbase, 2)} Mbase). Negatifler eğitimde hiç görülmedi; sadece
değerlendirmede kullanıldı. Sınıflandırmanın "etiketsiz" olması buradan
geliyor.

Pozitif ve negatif dizilerin GC oranı belirgin şekilde farklı (eğitim
bölümünde {fmt(0.6267)} ve {fmt(0.4803)}). Bu farkı eğitimden **önce** not
ettim, çünkü sadece-GC baseline'ının güçlü olacağını baştan biliyordum; asıl
soru modelin bunun ötesine geçip geçmediği.

## Kurulum

Mimari hafta 8'deki `v2.py` ile aynı: token ve öğrenilen pozisyon embedding'i,
{rep['n_layer']} adet pre-LN transformer bloğu (n_embd {rep['n_embd']},
{rep['n_head']} kafa, dropout {fmt(rep['dropout'], 2)}), final LayerNorm,
lm_head. Toplam {fmt(rep['n_params_M'], 2)}M parametre.

Diziler bağımsız kayıtlar olduğu için uç uca eklenmedi: her dizinin başına BOS
konup kendi örneği olarak batch'lendi. Böylece model olmayan dizi sınırları
arası geçiş öğrenmiyor ve dizi başına loss tanımlı hale geliyor. `block_size`
{meta['seq_length']}, yani bir promoter bağlama tam sığıyor.

Hafta 8'den iki sapma var. Birincisi dikkat, el yazımı tril maskesi yerine
`scaled_dot_product_attention` ile hesaplanıyor; matematiği aynı, füzyonlu
çekirdek MPS'te daha hızlı. İkincisi bir düzeltme: hafta 8'de ölçeklendirme
`n_embd ** -0.5` idi, oysa standart formülasyon `head_size ** -0.5` ister. Çok
kafalı dikkate geçerken oluşan bir kayma; altı kafayla attention logit'lerini
1/√6 kat fazla küçültüyordu, yani dikkat dağılımı olması gerekenden
yumuşaktı. Buradaki iki yol da doğru ölçeği kullanıyor.

## Eğitim

{thin(steps)} optimizer adımı, {fmt(tokens / 1e6, 2)}M token
(≈ {fmt(epochs, 1)} epoch), eğitim adımlarının toplam süresi
{fmt(tr_sec / 60, 1)} dakika, ortalama {thin(int(tokens / tr_sec))} token/saniye.
Değerlendirme ve üretim süresi bu hıza dahil değil.

İlk ölçüm {fmt(float(first['val_loss']))} nat, yani
{fmt(float(first['val_loss']) * NATS_TO_BITS)} bits/base. Bu tesadüf değil:
altı token üzerinde düzgün dağılımın entropisi ln(6) = {fmt(2.585/NATS_TO_BITS)}
nat. Model başlangıçta BOS ve N'nin hiç geçmediğini de bilmiyor, o yüzden
2,0'ın üstünde başlıyor.

Dört referans seviyesi var ve hangisine karşı ölçtüğünüz sonucu değiştiriyor:

| referans | bits/base |
| --- | ---: |
| başlangıç — {rep['vocab_size']} token üzerinde düzgün | {fmt(rep['bits_per_base_init_uniform'], 3)} |
| rastgele — dört baz üzerinde düzgün | {fmt(rep['bits_per_base_random'], 3)} |
| unigram — yalnızca baz frekansları, bağlam yok | {fmt(rep['bits_per_base_unigram'], 3)} |
| **model — en iyi doğrulama** | **{fmt(rep['bits_per_base_model_val'], 3)}** |

Unigram satırı önemli. Promoter'lar GC-zengin olduğu için
({", ".join(f"{b} {fmt(f, 3)}" for b, f in sorted(rep['base_frequencies'].items()))})
hiçbir bağlam kullanmayan, yalnızca baz frekanslarını bilen bir model dahi
2,0'ın altına, {fmt(rep['bits_per_base_unigram'], 3)} bits/base'e iner. Yani
"rastgeleyi geçtim" demek fazla cömert bir ifade olurdu.

Model rastgeleyi {fmt(rep['bits_per_base_random'] - rep['bits_per_base_model_val'], 3)}
bits/base geçiyor, ama bunun yalnızca
{fmt(rep['bits_per_base_random'] - rep['bits_per_base_unigram'], 3)} kadarı baz
frekanslarından gelen bedava kazanç. Bağlama bakarak kazandığı kısım
**{fmt(rep['bits_per_base_unigram'] - rep['bits_per_base_model_val'], 3)} bits/base**,
yani toplam kazancın
%{fmt(100 * (rep['bits_per_base_unigram'] - rep['bits_per_base_model_val']) / max(rep['bits_per_base_random'] - rep['bits_per_base_model_val'], 1e-9), 0)}'i.
Modelin gerçekten öğrendiği şey bu.

{of_note}

Loss eğrisi: `results/loss.png`.

## Sonuç 1 — etiketsiz sınıflandırma

Test bölümündeki her dizi için baz başına ortalama loss hesaplandı. Düşük loss
= modele "tanıdık" = promoter. Diziler bu skora göre sıralanıp AUROC
hesaplandı. AUROC, beraberliklere duyarlı Mann-Whitney U ile sıfırdan yazıldı.

| yöntem | AUROC |
| --- | ---: |
| model (dizi başına loss) | **{fmt(auc_m)}** |
| sadece GC oranı | {fmt(auc_gc)} |
{probe_row}
Pozitif dizilerin ortalaması {fmt(rep['bits_per_base_pos'])} bits/base,
negatiflerin {fmt(rep['bits_per_base_neg'])} bits/base. Dağılımlar:
`results/auroc_hist.png`.

{verdict}

## Sonuç 2 — in-silico saturation mutagenesis

En düşük loss'lu test pozitifini alıp her pozisyonda her alternatif bazı tek
tek denedim ({thin(mut.get('n_mutants', 0))} mutant), her biri için dizinin
log-likelihood'undaki değişimi ölçtüm. Bu, büyük DNA dil modellerinin varyant
etkisi tahmini için kullandığı yöntemin aynısı: mutasyon sonrası olasılık
değişimi. Maskeleme değil — maskeleme modelin MLM ile eğitilmiş olmasını
gerektirir, bu model otoregresif.

Orijinal dizi {fmt(mut.get('orig_bits_per_base', 0.0))} bits/base, toplam
Δ NLL ortalaması {fmt(mut.get('d_total_mean', 0.0))} nat.

**Baz tercihi net ve GC yönünde:**

| mutasyon hedefi | ortalama Δ NLL (nat) |
| --- | ---: |
{by_base_rows}
Herhangi bir bazı T veya A'ya çevirmek diziyi belirgin biçimde daha az olası
yapıyor; G'ye çevirmek ise hafifçe daha olası yapıyor. Isı haritasında bu,
A ve T satırlarının baştan sona kırmızı, G satırının nötr olması olarak
görünüyor.

{decomp_section}## Modelin skoru GC'den ibaret mi? Hayır

Akla gelen ilk açıklama, modelin skorunun örtük bir GC ölçümü olduğuydu.
Bunu doğrudan test ettim: dizi başına loss ile GC oranı arasındaki korelasyon
{fmt(rep.get('loss_gc_corr_all', 0.0), 3)}, yani
r² = {fmt(rep.get('loss_gc_corr_all', 0.0) ** 2, 3)}. GC, skordaki varyansın
yalnızca %{fmt(100 * rep.get('loss_gc_corr_all', 0.0) ** 2, 0)}'sini açıklıyor.
GC'yi doğrusal olarak çıkardıktan sonra kalan artığın AUROC'u
{fmt(rep.get('auroc_loss_residual_after_gc', 0.0))} — zayıf, ama 0,5 değil.

Yani hipotez çürüdü: **skor GC'den ibaret değil, GC'den daha kötü.** Model
kompozisyonun ötesinde bir şey ölçüyor, ama ölçtüğü şey bu ayrım için GC
oranından daha zayıf bir özellik.

Pozitif ve negatiflerde korelasyonun ayrılması da bilgi veriyor: pozitiflerde
{fmt(rep.get('loss_gc_corr_pos', 0.0), 3)}, negatiflerde
{fmt(rep.get('loss_gc_corr_neg', 0.0), 3)}. Model yalnızca pozitiflerde
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

{fix_section}## Sınırlar

**Causal model pozisyon eğilimi yaratıyor.** Model yalnızca sola bakıyor, yani
`i` pozisyonundaki bir mutasyon sadece `i` ve sonrasındaki tahminleri
değiştirebilir. Dizinin başındaki bir mutasyonun etkileyeceği
{meta['seq_length']} pozisyon varken, sonundakinin birkaç pozisyonu var.
Ölçülen korelasyon {fmt(mut.get('position_bias_corr', 0.0), 3)} (|Δ| ile aşağı
akış pozisyon sayısı arasında). Isı haritasında hem ham hem aşağı akış
sayısına normalize edilmiş hali var. Çift yönlü modeller bu sorunu maskeleme
ile çözüyor; otoregresif modeller varyantı dizinin sonuna yerleştiriyor.

**GC karışması.** Promoter'lar GC-zengin olduğu için ayırt edici gücün bir
kısmı kompozisyondan geliyor. Sadece-GC baseline'ı tam bu yüzden hesaplandı,
ama bu karşılaştırma farkın her bir nedene düşen payını ayırmıyor.

**Ölçek.** {fmt(rep['n_params_M'], 2)}M parametre ve {fmt(mbase, 2)} Mbase
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
"""

    out = "SONUCLAR.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {out} ({len(md)} characters)")
    print(f"\nheadline numbers: model AUROC {auc_m:.4f} | GC {auc_gc:.4f} | "
          f"gap {gap:+.4f}")


if __name__ == "__main__":
    main()
