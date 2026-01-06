from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import time
import random
import numpy as np
import re
import os
import seaborn as sns
import matplotlib.pyplot as plt


def setup_driver():
    chrome_options = Options()

    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    chrome_options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')

    #chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')

    chrome_options.add_argument('--disable-web-security')
    chrome_options.add_argument('--allow-running-insecure-content')

    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
    }
    chrome_options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=chrome_options)

    driver.implicitly_wait(10)

    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
    })

    return driver


def get_product_specifications(driver, product_url, max_retries=2):
    for attempt in range(max_retries):
        try:
            driver.get(product_url)
            time.sleep(random.uniform(2, 4))

            if "trafic neobișnuit" in driver.page_source.lower():
                return {}

            specs_container = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "specifications-body"))
            )

            driver.execute_script(
                "arguments[0].scrollIntoView({block:'start'});",
                specs_container
            )
            time.sleep(1)

            target_categories = {
                "alimentare",
                "afișare", "afisare",
                "memorie"
            }

            all_tables = specs_container.find_elements(
                By.CSS_SELECTOR,
                "table.specifications-table, table.table-striped"
            )

            all_p = specs_container.find_elements(
                By.XPATH, ".//p[contains(@class,'text-uppercase')]"
            )

            table_categories = []

            for table in all_tables:
                cat_name = None

                for p_elem in all_p:
                    try:
                        next_table = p_elem.find_element(
                            By.XPATH,
                            "following::table[contains(@class,'specifications-table')][1]"
                        )

                        if next_table == table:
                            cat_name = (
                                p_elem.get_attribute("textContent")
                                or p_elem.get_attribute("innerText")
                                or p_elem.text
                            )
                            if cat_name:
                                cat_name = cat_name.strip()
                            break
                    except:
                        continue

                table_categories.append(cat_name)

            specs = {}

            for table, category in zip(all_tables, table_categories):
                if not category:
                    continue

                if category.lower() not in target_categories:
                    continue

                rows = table.find_elements(By.TAG_NAME, "tr")

                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")

                    if len(cells) >= 2:
                        key = (
                            cells[0].get_attribute("textContent")
                            or cells[0].get_attribute("innerText")
                            or cells[0].text
                        )
                        value = (
                            cells[1].get_attribute("textContent")
                            or cells[1].get_attribute("innerText")
                            or cells[1].text
                        )

                        key = key.strip() if key else ""
                        value = value.strip() if value else ""

                        if key and value:
                            specs[f"{category} - {key}"] = value

            return specs

        except Exception:
            if attempt < max_retries - 1:
                time.sleep(random.uniform(2, 4))
                continue
            return {}

    return {}



def parse_product_card(card, max_retries=2):
    for attempt in range(max_retries):
        try:
            product_data = {}

            try:
                title_elem = card.find_element(By.CSS_SELECTOR, "a.card-v2-title, a[class*='title']")
                product_data['title'] = title_elem.text.strip()
                product_data['url'] = title_elem.get_attribute('href')
            except (NoSuchElementException, Exception):
                product_data['title'] = ""
                product_data['url'] = ""

            try:
                price_elem = card.find_element(By.CSS_SELECTOR, "p.product-new-price, p[class*='price']")
                price_text = price_elem.text.strip()
                price_clean = price_text.replace('lei', '').replace('RON', '').replace('.', '').replace(',',
                                                                                                        '.').strip()
                product_data['price_raw'] = price_text
                product_data['price_clean'] = price_clean
            except (NoSuchElementException, Exception):
                product_data['price_raw'] = ""
                product_data['price_clean'] = ""

            try:
                rating_elem = card.find_element(By.CSS_SELECTOR, "[class*='rating']")
                product_data['rating'] = rating_elem.get_attribute('data-rating') or rating_elem.text
            except (NoSuchElementException, Exception):
                product_data['rating'] = ""

            try:
                img_elem = card.find_element(By.CSS_SELECTOR, "img")
                product_data['image'] = img_elem.get_attribute('src') or img_elem.get_attribute('data-src')
            except (NoSuchElementException, Exception):
                product_data['image'] = ""

            return product_data if product_data.get('title') else None

        except Exception as e:
            if "stale element" in str(e) and attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            return None

    return None


def scrape_emag_page(driver, url, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Accesez: {url} (încercare {attempt + 1}/{max_retries})")
            driver.get(url)

            wait = WebDriverWait(driver, 15)

            selectors = [
                "div.card-item",
                "div[class*='card-item']",
                "div.card-v2",
                "div[class*='product-card']",
                "div.card"
            ]

            cards = []
            for selector in selectors:
                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                    cards = driver.find_elements(By.CSS_SELECTOR, selector)
                    if cards:
                        print(f"✓ Găsite {len(cards)} produse cu selectorul: {selector}")
                        break
                except TimeoutException:
                    continue

            if not cards:
                print("⚠ Nu s-au găsit produse cu selectorii disponibili")
                # Salvează screenshot pentru debugging
                driver.save_screenshot(f"debug_page_{int(time.time())}.png")
                return []

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            for selector in selectors:
                try:
                    cards = driver.find_elements(By.CSS_SELECTOR, selector)
                    if cards:
                        break
                except:
                    continue

            products = []
            for i, card in enumerate(cards):
                try:
                    product = parse_product_card(card)
                    if product:
                        products.append(product)
                        # Afișează detalii despre produs
                        print(f"  [{i + 1}] {product['title'][:80]}")
                        print(f"      💰 Preț: {product['price_raw']}")
                        if product.get('rating'):
                            print(f"      ⭐ Rating: {product['rating']}")
                        print(f"      🔗 {product['url'][:100]}")
                        print()
                except Exception as e:
                    continue

            return products

        except Exception as e:
            print(f"Eroare la scraping pagină: {e}")
            if attempt < max_retries - 1:
                print(f"Reîncerc după {2 ** attempt} secunde...")
                time.sleep(2 ** attempt)
            else:
                print("S-au epuizat reîncercările")
                return []

    return []


def scrape_emag(search_query="telefoane", max_pages=30, output_csv="emag_products.csv", get_specs=True):
    driver = setup_driver()
    all_products = []

    try:
        for page_num in range(1, max_pages + 1):
            url = f"https://www.emag.ro/search/{search_query}/p{page_num}"

            products = scrape_emag_page(driver, url)

            if not products:
                print(f"Pagina {page_num} nu are produse - se oprește scraping-ul")
                break

            if get_specs:
                print(f"\n🔍 Extrag specificații pentru {len(products)} produse...\n")
                for idx, product in enumerate(products, 1):
                    print(f"  📱 [{idx}/{len(products)}] Extrag specs pentru: {product['title'][:60]}...")

                    if product.get('url'):
                        specs = get_product_specifications(driver, product['url'])

                        # Verifică dacă am fost blocați
                        if not specs and "trafic" in driver.page_source.lower():
                            print("\n❌ eMAG ne-a blocat! Opresc scraping-ul.")
                            print("💡 Recomandări:")
                            print("   1. Așteaptă 30-60 minute")
                            print("   2. Schimbă IP-ul (restart router sau VPN)")
                            print("   3. Rulează cu get_specs=False (mai puțin intrusiv)")
                            print("   4. Crește delay-urile în cod")
                            break

                        for key, value in specs.items():
                            product[f"spec_{key}"] = value

                        if specs:
                            print(f"      ✓ Găsite {len(specs)} specificații")
                            for i, (key, value) in enumerate(list(specs.items())):
                                print(f"        • {key}: {value[:60]}")
                        else:
                            print(f"      ⚠ Nu s-au găsit specificații")

                        print()

                        delay = random.uniform(2, 4)
                        print(f"      ⏳ Pauză {delay:.1f} secunde...")
                        time.sleep(delay)

                    driver.back()
                    time.sleep(random.uniform(2, 4))

            all_products.extend(products)
            print(f"\n{'=' * 80}")
            print(f"📊 Total produse până acum: {len(all_products)}")
            print(f"{'=' * 80}\n")

            # Delay MULT mai mare între pagini
            delay = random.uniform(2, 4)
            print(f"⏳ Pauză {delay:.1f} secunde înainte de următoarea pagină...\n")
            time.sleep(delay)

    except KeyboardInterrupt:
        print("\n⚠ Scraping întrerupt de utilizator")

    finally:
        driver.quit()
        print("Browser închis")

    if all_products:
        df = pd.DataFrame(all_products)

        df = df.drop_duplicates(subset=['url'], keep='first')

        basic_cols = ['title', 'price_raw', 'price_clean', 'rating', 'url', 'image']
        spec_cols = [col for col in df.columns if col.startswith('spec_')]

        ordered_cols = [col for col in basic_cols if col in df.columns] + sorted(spec_cols)
        df = df[ordered_cols]

        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n✅ {len(df)} produse salvate în '{output_csv}'")

        print(f"\n{'=' * 80}")
        print(f"📊 STATISTICI FINALE")
        print(f"{'=' * 80}")
        print(f"Total produse unice: {len(df)}")
        print(f"Coloane în CSV: {len(df.columns)}")
        print(f"Specificații extrase: {len(spec_cols)}")
        print(f"{'=' * 80}\n")

        print(f"📋 PRIMELE 3 PRODUSE (cu specificații):\n")
        for idx in range(min(3, len(df))):
            row = df.iloc[idx]
            print(f"{'=' * 80}")
            print(f"🔢 PRODUS #{idx + 1}")
            print(f"{'=' * 80}")
            print(f"📱 Titlu: {row['title']}")
            print(f"💰 Preț: {row['price_raw']}")
            if 'rating' in row and pd.notna(row['rating']) and row['rating']:
                print(f"⭐ Rating: {row['rating']}")
            print(f"🔗 URL: {row['url']}\n")

            specs_found = False
            print("📋 SPECIFICAȚII:")
            for col in df.columns:
                if col.startswith('spec_') and pd.notna(row[col]) and row[col]:
                    specs_found = True
                    spec_name = col.replace('spec_', '')
                    spec_value = str(row[col])[:100]
                    print(f"   • {spec_name}: {spec_value}")

            if not specs_found:
                print("   ⚠ Nu s-au găsit specificații pentru acest produs")

            print()

        print(f"{'=' * 80}")
        print(f"📄 PREVIEW CSV (primele 5 rânduri):")
        print(f"{'=' * 80}\n")

        preview_cols = ['title', 'price_raw', 'rating']
        preview_cols += [col for col in spec_cols[:5] if col in df.columns]

        preview_df = df[preview_cols].head(5)

        preview_display = preview_df.copy()
        for col in preview_display.columns:
            preview_display[col] = preview_display[col].astype(str).str[:50]

        print(preview_display.to_string(index=True))
        print(f"\n{'=' * 80}")
        print(f"💾 CSV complet salvat în: {output_csv}")
        print(f"{'=' * 80}\n")

    else:
        print("\n❌ Nu s-au găsit produse de salvat")

def extract_brand(title):
    """Extrage brand-ul din titlu"""
    if not isinstance(title, str):
        return None

    title_lower = title.lower()

    # Caută fiecare brand în titlu
    for brand in brands:
        if brand in title_lower:
            return brand.capitalize()  # Returnează cu prima literă mare

    return None  # Dacă nu găsește niciun brand


def convert_memory_to_gb(text):
    if not isinstance(text, str):
        return np.nan

    gb = re.search(r'(\d+)\s?GB', text, re.IGNORECASE)
    if gb:
        return float(gb.group(1))

    mb = re.search(r'(\d+)\s?MB', text, re.IGNORECASE)
    if mb:
        return float(mb.group(1)) / 1024

    return np.nan


def calculate_megapixels(resolution):
    """Calculează megapixelii din rezoluție (ex: 720 x 1612 → 1.16 MP)"""
    if not isinstance(resolution, str):
        return np.nan

    # Extrage width x height
    match = re.search(r'(\d+)\s?x\s?(\d+)', resolution, re.IGNORECASE)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        # Calculează megapixeli
        megapixels = (width * height) / 1_000_000
        return round(megapixels, 2)

    return np.nan

def clean_price(text):
    if isinstance(text, str):
        text = text.replace("Lei", "").replace("RON", "").replace("lei", "").strip()

        text = text.replace(".", "")

        text = text.replace(",", ".")

        nums = re.findall(r"[\d.]+", text)
        if nums:
            try:
                return float(nums[0])
            except ValueError:
                return np.nan
    return np.nan


if __name__ == "__main__":
    # nu trebuie rulata, dureaza foarte mult scrapping-ul
    # scrape_emag(search_query="telefoane", max_pages=80, output_csv="../data/emag_phones.csv", get_specs=True)
    brands = [
        "agm", "aiek", "aiwa", "alcatel", "alexverity", "allview", "apple", "asus",
        "black-fox", "blackview", "blasko", "blaupunkt", "cat", "cmf", "cmf-by-nothing",
        "coolpad", "cubot", "czay", "doogee", "e-boda", "eaor", "energizer", "evolveo",
        "f150", "fairphone", "fossibot", "foxmag24", "gadgetx", "gaia", "gamakoo",
        "gegeszoft", "gigaset", "google", "hafury", "hammer", "hmd", "homtom", "honor",
        "hotwav", "huawei", "ihunt", "infinix", "ipro", "isen", "itpro", "izowe",
        "kruger-matz", "l8-star", "l8star", "lagenio", "maxcom", "meizu", "mobiola",
        "motorola", "mrg", "my-phone", "myphone", "myria", "neohgs", "nokia", "norton",
        "nothing", "nubia", "oem", "one", "oneplus", "opis-technology", "oppo", "orange",
        "oscal", "oukitel", "panasonic", "poco", "powertech", "qubo", "rainbuvvy",
        "realme", "redmagic", "redmi", "rugged", "rugone", "samsung", "smartgadget",
        "sol", "sonim", "sony", "soyes", "spc", "stk", "syno", "tcl", "tecno", "trevi",
        "ulefone", "umidigi", "unihertz", "universal", "uniwa", "vivo", "watchiu",
        "xiaomi", "yeemi", "zte"
    ]
    df = pd.read_csv("../data/emag_phones.csv")
    df.head()
    df.info()

    df["price"] = df["price_raw"].apply(clean_price)

    # Vérifier
    print(df[["title", "price_raw", "price"]].head(10))
    print("Valeurs manquantes price:", df["price"].isna().sum())

    # df["storage"] = df["title"].str.extract(r'(\d+)\s?GB')[0].astype(float)
    df["storage"] = df["spec_Memorie - Memorie RAM"].apply(convert_memory_to_gb)
    df["ram"] = df["spec_Memorie - Memorie interna"].apply(convert_memory_to_gb)

    # Remplir les valeurs manquantes par la médiane (optionnel)
    df["storage"] = df["storage"].fillna(df["storage"].median())
    df["ram"] = df["ram"].fillna(df["ram"].median())

    df["brand"] = df["title"].apply(extract_brand)
    df["battery"] = df["spec_Alimentare - Capacitate baterie"].str.extract(r'(\d+)\s?mAh')[0].astype(float)
    df["screen_megapixels"] = df["spec_Afisare - Rezolutie (pixeli)"].apply(calculate_megapixels)
    df["screen_size"] = (
        df["spec_Afisare - Dimensiune ecran"]
        .str.extract(r'(\d+(?:[.,]\d+)?)')[0]
        .str.replace(',', '.', regex=False)
        .astype(float)
        .round(2)
    )
    df["charger"] = (
        df["spec_Alimentare - Incarcator inclus"]
        .str.lower()
        .map({
            "da": 1,
            "da, incarcare rapida usb pd": 1,
            "yes": 1,
            "inclus": 1,
            "nu": 0,
            "no": 0
        })
    )
    df["charger"] = df["charger"].fillna(df["charger"].median()).astype(int)

    df["rating"] = (
        df["rating"]
        .str.extract(r'(\d+(?:\.\d+)?)')[0]
        .astype(float)
    )
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df_ml = df[["price", "storage", "ram", "brand", "battery", "screen_megapixels", "screen_size", "charger",
                "rating"]].dropna()

    # One-Hot Encoding pour brand
    df_ml = pd.get_dummies(df_ml, columns=["brand"], drop_first=True)

    df_ml.head()
    df_ml.info()
    print(df_ml["rating"].head(10))

    print(os.getcwd())
    df_ml.to_csv("../data/emag_phones_ml_ready.csv", index=False, encoding="utf-8-sig")
    print("✅ Fichier enregistré dans le dossier courant.")

    print(df_ml.describe())
