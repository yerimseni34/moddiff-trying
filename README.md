# texproj — referans stilde 3D model texture'lama (render → GPT → projeksiyon)

Blender'a hiç girmeden, tek komutla:

1. **render** – modeli 6–10 sabit ortografik açıdan (ön/arka/sağ/sol/üst/alt…) render eder, hepsini tek bir grid `sheet.png` içine de dizer
2. **stylize** – render'ları + senin stil referans görsellerini GPT image API'ye yollar, aynı stilde boyatır
3. **project** – boyanmış görselleri *birebir aynı kameralarla* modele geri projekte edip UV texture'a bake eder
4. **fix** – hiçbir açının göremediği bölgeleri bulur, tam oraya bakan ekstra açılardan render alıp AI'a sadece o bölgeleri boyatır

Çıktı: `out/texture.png`, `out/model.obj` (+mtl), `out/model.glb`, `out/preview.png`.

## Arayüz

```bash
texproj ui
```

Tarayıcıda açılır: modeli yükle → **Render al** → `sheet.png`'yi indirip (hazır prompt'la birlikte)
ChatGPT'de boyat → boyanmış sheet'i yükle → **Projekte et**. Sonuç 3D görüntüleyicide ve 8 açılı
önizlemede görünür; texture / GLB / OBJ indirilebilir. 3. adım görülmeyen yerler için düzeltme
açıları üretir. API anahtarın varsa "API ile (otomatik)" sekmesi boyamayı da kendi yapar.

## Windows'ta hızlı başlangıç

1. [python.org](https://www.python.org/downloads/) üzerinden **Python 3.12** kur
   (kurulumda **"Add python.exe to PATH"** kutusunu işaretle).
2. Bu repoyu indir (GitHub'da branch'i seç → **Code → Download ZIP**) ve bir klasöre çıkar.
3. **`kur.bat`**'a çift tıkla (bir kere, birkaç dakika sürer).
4. **`baslat.bat`**'a çift tıkla → arayüz tarayıcıda açılır (http://127.0.0.1:7860).

## Kurulum

```bash
pip install -e .            # numpy, pillow, scipy, trimesh, xatlas, openai, gradio
export OPENAI_API_KEY=sk-...
```

GPU / OpenGL / Blender gerekmez (render ve bake saf numpy).

## Kullanım

Hepsi tek seferde:

```bash
texproj run karakter.glb -o work --ref stil1.png stil2.png \
    --prompt "stylized fantasy knight, hand-painted, metal armor, red cape" \
    --views 6 --mode hybrid --tex 2048
```

Adım adım (her adımın çıktısı `work/` klasöründe durur, istediğin adımı tekrar edebilirsin):

```bash
texproj render  karakter.glb -o work --views 8
texproj stylize work --ref stil.png --mode grid
texproj project work --tex 2048
texproj fix     work --ref stil.png          # opsiyonel: delikleri AI ile kapat
```

### API'siz (ChatGPT'de elle) kullanım

```bash
texproj render karakter.glb -o work
```

`work/sheet.png` (tüm açılar tek görselde) veya `work/views/*.png` dosyalarını ChatGPT'ye stil
referansınla birlikte ver, sonucu **aynı isimle** `work/stylized/` içine koy
(`stylized/sheet.png` *veya* `stylized/front.png, right.png, …`), sonra:

```bash
texproj project work
texproj fix work --render-only   # fix/views + fix/masks üretir; boyayıp fix/stylized/ içine koy
texproj project work
```

Bir açı kötü çıktıysa sadece o dosyayı yeniden ürettirip `project`'i tekrar çalıştırman yeterli.

### Stylize modları

| mod | çağrı sayısı | ne yapar |
|---|---|---|
| `grid` (varsayılan) | 1 | Tüm açılar tek görselde boyanır → açılar arası tutarlılık en yüksek, çözünürlük hücre başına düşük (`--sheet-cell`) |
| `per-view` | N | Her açı ayrı, tam çözünürlükte; önceki boyanmış açılar referans olarak verilir |
| `hybrid` | N+1 | Önce grid (tutarlılık), sonra her açı grid'deki hâli rehber alınarak tam çözünürlükte yeniden boyanır. **En iyi kalite** |

`--image-model` ile model seçilir (varsayılan `gpt-image-2`; `gpt-image-1.5` vb. de olur).

## Neden Blender'daki projection araçlarından daha iyi sonuç verir

- **Texel bazlı bake**: her texel'in 3D konumu/normali bilinir; her açı için derinlik testi yapılır →
  arkadaki yüzeye "boya sızması" olmaz.
- **Açı + kenar ağırlığı**: bir piksel, yüzeye ne kadar dik bakıyorsa o kadar ağırlık alır; siluet ve
  oklüzyon kenarlarına yakın pikseller (AI'ın en çok saçmaladığı yerler) yumuşakça devre dışı kalır.
  Her texel pratikte onu en iyi gören açıdan boyanır (`--sharp`).
- **Otomatik hizalama**: image modelleri objeyi birkaç piksel kaydırır/büyütür. Boyanmış görselin
  silüeti render maskesine otomatik oturtulur ve her açı için IoU raporlanır; şekli bozulan açı
  (`IoU < 0.9`) uyarı olarak işaretlenir → o açıyı yeniden ürettir.
- **Renk eşitleme**: açılar arası parlaklık/ton farkı örtüşen bölgelerden hesaplanıp dengelenir.
- **Delik doldurma 3D'de**: görülmeyen texel'ler UV uzayında değil, 3D'deki en yakın boyalı
  komşulardan doldurulur (UV adaları arası renk taşması olmaz); istersen `fix` ile AI doldurur.
- `out/debug/holes.png` ve `confidence.png` hangi bölgelerin zayıf kaldığını gösterir.

## İpuçları

- Model **Y-up, önü +Z** kabul edilir (Blender'ın varsayılan OBJ/GLB export'u böyle). Ham Blender
  koordinatı için `--up z`, ön yüz yanlış tarafa bakıyorsa `--yaw 90/180/270`.
- UV'si olmayan modeller otomatik unwrap edilir (xatlas). **Mirror/üst üste binen UV** varsa
  projeksiyon doğru tutulamaz; araç uyarır → `--unwrap` ile temiz UV açtır.
- Çok parçalı/girintili modellerde `--views 8` veya `10` kullan; kalan yerleri `fix` kapatır.
- `--prompt` içine objenin ne olduğunu ve malzemeleri yaz; stil referansı ne kadar netse sonuç o
  kadar tutarlı olur.

## Test (API'siz)

```bash
python tests/selftest.py
```

Sahte bir "stil" (bilinen bir 3D desen) kullanıp, her açıyı kasıtlı olarak kaydırıp/ölçekleyip
tüm pipeline'ı çalıştırır ve bake edilen texture'ı gerçek desenle karşılaştırır.
