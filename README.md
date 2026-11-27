# texproj — referans stilde 3D model texture'lama (render → GPT → projeksiyon)

Blender'a hiç girmeden, tek komutla:

1. **render** – modeli 6–10 sabit ortografik açıdan (ön/arka/sağ/sol/üst/alt…) render eder, hepsini tek bir grid `sheet.png` içine de dizer
2. **stylize** – render'ları + senin stil referans görsellerini GPT image API'ye yollar, aynı stilde boyatır
3. **project** – boyanmış görselleri *birebir aynı kameralarla* modele geri projekte edip UV texture'a bake eder
4. **fix** – hiçbir açının göremediği bölgeleri bulur, tam oraya bakan ekstra açılardan render alıp AI'a sadece o bölgeleri boyatır

Çıktı: `out/texture.png`, `out/model.obj` (+mtl), `out/model.glb`, `out/preview.png`.

## Arayüz

```bash
texproj ui        # Windows'ta: baslat.bat
```

1. **Model:** dosyayı yükle → **Görselleri hazırla**. Boyatılacak açılar, hepsini içeren bir zip ve
   ChatGPT prompt'u çıkar.
2. **Boyanmış görselleri yükle:** ChatGPT'den gelenleri hepsini birden sürükle. İsimleri ve sırası
   önemli değil; program siluetlerine bakıp hangi açıya ait olduklarını kendisi bulur. Yanlış eşleşen
   olursa o açının kutusuna doğrusunu koy. → **Çalıştır**.
3. **Sonuç:** texture'sız ve texture'lı model yan yana, döndürülebilir 3D görüntüleyicide; altında
   8 açılı önizleme ve indirilecek dosyalar (GLB, OBJ, texture.png).
4. **AI ile düzelt (yerel):** düzeltilmesi gereken yerler modelin üstünde kırmızıyla gösterilir.
   Butona basınca program bu yerlere yakından bakan ekstra açılar çeker, kendi ekran kartında çalışan
   SDXL ile **sadece bu yerleri** senin boyattığın görselin stilinde yeniden boyar ve modele geri
   yansıtır. Önce/sonra görüntüleri de gösterilir.

## Görseller neden yakınlaştırılmış

Her açı modele sıkıca oturan kendi çerçevesiyle render edilir; uzun figürler için dikey (1024×1536),
geniş görünüşler için yatay (1536×1024) format seçilir (ChatGPT'nin doğal oranları). Böylece
görselin boş beyaz alanı küçülür ve aynı piksel sayısıyla modele çok daha fazla detay düşer
(test modelinde ön/yan/arka 1.5 kat, üst/alt 2.7 kat).

## Yerel AI düzeltme (API ve abonelik gerektirmez)

Windows'ta bir kere **`kur_ai.bat`** çalıştır (PyTorch'un CUDA 12.8 sürümü + diffusers; RTX 50xx
için bu sürüm gerekli). İlk "AI ile düzelt"te modeller indirilir (~10 GB: SDXL inpainting +
IP-Adapter), sonraki çalıştırmalar bunları kullanır. 12 GB VRAM yeterli.

Düzeltilecek yerler üç şeyden hesaplanır, ama sadece **bütün halinde** kötü olan bölgeler sayılır
(kıvrımlardaki ince çizgilere dokunulmaz):

- hiçbir açının görmediği yerler,
- sadece çok yandan / zayıf görülen yerler,
- birden fazla açının karıştığı ve birbiriyle çeliştiği yerler (dikiş izi / gölge kayması).

AI maskenin dışındaki hiçbir pikseli değiştirmez; projeksiyon da sadece maskelenen pikselleri kullanır.
Stil için IP-Adapter'a senin boyattığın ön görsel verilir. Ayarlar: düzeltme açısı sayısı, yeniden
boyama gücü, stil referansı gücü; model alanına başka bir SDXL modeli (HF id ya da `.safetensors`
yolu, örn. anime odaklı bir SDXL checkpoint) yazılabilir.

Komut satırından: `texproj fix work --backend local --prompt "anime girl, blue hair"`.

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
