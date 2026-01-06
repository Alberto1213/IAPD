"""
Script de TEST pentru a vedea exact ce specificații găsește pe o pagină eMAG
Extrage DOAR specificațiile de Alimentare și Afișare
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time


def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')

    # COMENTEAZĂ linia de mai jos pentru a VEDEA browser-ul în timp real
    # chrome_options.add_argument('--headless=new')

    driver = webdriver.Chrome(options=chrome_options)
    driver.implicitly_wait(10)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    return driver


def test_specifications(product_url):
    """Testează extragerea de specificații pe un produs - DOAR Alimentare și Afișare"""
    driver = setup_driver()

    try:
        print(f"🔗 Accesez: {product_url}\n")
        driver.get(product_url)
        time.sleep(3)

        # Verificare blocare
        if "trafic neobișnuit" in driver.page_source.lower():
            print("❌ Ești blocat de eMAG!")
            return

        print("=" * 80)
        print("EXTRAGERE SPECIFICAȚII (Alimentare + Afișare)")
        print("=" * 80)

        # Scroll la specificații
        try:
            specs_section = driver.find_element(By.ID, "specifications-body")
            driver.execute_script("arguments[0].scrollIntoView(true);", specs_section)
            time.sleep(2)
            print("✓ Am găsit div#specifications-body\n")
        except:
            print("✗ Nu am găsit div#specifications-body\n")

        # FILTRARE: Lista de categorii dorite
        target_categories = [
            "alimentare", "afișare", "afisare", "display",
            "ecran", "baterie", "battery", "autonomie"
        ]
        print(f"🎯 Categorii căutate: {', '.join(target_categories)}\n")

        # Găsește toate tabelele de specificații
        all_tables = driver.find_elements(By.CSS_SELECTOR, "table.specifications-table, table.table-striped")
        print(f"📋 Total tabele găsite: {len(all_tables)}\n")

        # Mai întâi, afișează TOATE categoriile disponibile
        print("📂 GĂSIRE CATEGORII:\n")

        # Debug: Găsește toate <p> cu categorii
        try:
            specs_container = driver.find_element(By.ID, "specifications-body")
            all_category_p = specs_container.find_elements(By.XPATH, ".//p[contains(@class, 'text-uppercase')]")
            print(f"🔍 DEBUG: Găsite {len(all_category_p)} paragrafe <p> cu text-uppercase\n")

            for p_idx, p_elem in enumerate(all_category_p, 1):
                # Afișează HTML-ul complet al <p>
                p_html = p_elem.get_attribute('outerHTML')
                print(f"    Paragraf {p_idx}:")
                print(f"      HTML: {p_html[:200]}")

                # Încearcă multiple metode de extragere text
                cat_text = ""

                # Metodă 1: textContent (funcționează și pentru elemente ascunse)
                cat_text = p_elem.get_attribute('textContent')
                if cat_text:
                    cat_text = cat_text.strip()

                # Metodă 2: innerText (fallback)
                if not cat_text:
                    cat_text = p_elem.get_attribute('innerText')
                    if cat_text:
                        cat_text = cat_text.strip()

                # Metodă 3: .text (fallback final)
                if not cat_text:
                    cat_text = p_elem.text.strip()

                if cat_text:
                    print(f"      Text: '{cat_text}'")
                else:
                    print(f"      Text: (GOL)")

                print()

            print()
        except Exception as e:
            print(f"⚠ Eroare la găsire categorii: {e}\n")

        print("📋 ASOCIERE TABELE ↔ CATEGORII:\n")
        all_categories = []

        for idx, table in enumerate(all_tables, 1):
            cat_name = None

            try:
                specs_container = driver.find_element(By.ID, "specifications-body")
                all_p = specs_container.find_elements(By.XPATH, ".//p[contains(@class, 'text-uppercase')]")

                # Pentru fiecare <p>, găsește următorul tabel
                for p_elem in all_p:
                    try:
                        # Găsește următorul tabel după acest <p>
                        next_table = p_elem.find_element(By.XPATH,
                                                         "following::table[@class='table table-striped specifications-table'][1]")

                        # Compară dacă e același tabel
                        if next_table.id == table.id or next_table == table:
                            # Extrage text folosind textContent (funcționează și pt elemente ascunse)
                            cat_name = p_elem.get_attribute('textContent')
                            if not cat_name:
                                cat_name = p_elem.get_attribute('innerText')
                            if not cat_name:
                                cat_name = p_elem.text

                            if cat_name:
                                cat_name = cat_name.strip()
                            break
                    except:
                        continue

            except Exception as e:
                print(f"  [{idx}] Eroare la căutare: {e}")

            if cat_name:
                all_categories.append(cat_name)
                print(f"  [{idx}] ✓ {cat_name}")
            else:
                all_categories.append(None)
                print(f"  [{idx}] ✗ (categoria neidentificată)")

        print(f"\n{'=' * 80}\n")

        # Acum procesează doar categoriile dorite
        all_specs = {}
        tables_processed = 0

        print("🔍 PROCESARE CATEGORII DORITE:\n")

        for idx, table in enumerate(all_tables, 1):
            try:
                category = all_categories[idx - 1]

                if not category:
                    continue

                # Verifică dacă e categorie dorită (case-insensitive)
                is_target = category.lower() in target_categories

                if not is_target:
                    print(f"  ⏭️  [{idx}] {category} - SKIP")
                    continue

                print(f"  ✅ [{idx}] {category} - PROCESEZ")
                tables_processed += 1

                # Extrage rânduri
                rows = table.find_elements(By.TAG_NAME, "tr")
                print(f"      → {len(rows)} rânduri în tabel")

                rows_extracted = 0
                for row_idx, row in enumerate(rows, 1):
                    cells = row.find_elements(By.TAG_NAME, "td")

                    print(f"        Rând {row_idx}: {len(cells)} celule")

                    if len(cells) >= 2:
                        # Folosește textContent pentru a extrage text chiar dacă e ascuns
                        key = cells[0].get_attribute('textContent')
                        value = cells[1].get_attribute('textContent')

                        if not key:
                            key = cells[0].get_attribute('innerText')
                        if not value:
                            value = cells[1].get_attribute('innerText')
                        if not key:
                            key = cells[0].text
                        if not value:
                            value = cells[1].text

                        key = key.strip() if key else ""
                        value = value.strip() if value else ""

                        print(f"          Key: '{key[:50]}'")
                        print(f"          Value: '{value[:50]}'")

                        # Skip rânduri goale
                        if key and value:
                            full_key = f"{category} - {key}"
                            all_specs[full_key] = value
                            rows_extracted += 1
                            print(f"          ✓ Salvat!")
                        else:
                            print(f"          ✗ Gol, skip")
                    else:
                        print(f"          ✗ Nu are 2 celule")

                if rows_extracted == 0:
                    print(f"        ⚠ Niciun rând valid găsit")
                else:
                    print(f"        ✓ {rows_extracted} rânduri extrase")

                print()

            except Exception as e:
                print(f"  ✗ Eroare la tabel {idx}: {e}\n")

        print("=" * 80)
        print(f"📊 REZULTATE FINALE:")
        print("=" * 80)
        print(f"Total tabele pe pagină: {len(all_tables)}")
        print(f"Categorii disponibile: {len([c for c in all_categories if c])}")
        print(f"Tabele procesate (categorii dorite): {tables_processed}")
        print(f"Specificații extrase: {len(all_specs)}")
        print("=" * 80)

        if all_specs:
            print(f"\n✅ SPECIFICAȚII EXTRASE:\n")
            for key, value in all_specs.items():
                print(f"  • {key}: {value}")
        else:
            print(f"\n⚠️ NU S-AU GĂSIT SPECIFICAȚII PENTRU CATEGORIILE DORITE!")
            print(f"\n💡 Sugestie: Verifică dacă produsul are categoriile:")
            for cat in target_categories:
                print(f"   - {cat.title()}")

        print(f"\n{'=' * 80}")

        # Salvează HTML pentru debugging dacă e necesar
        with open("debug_product_page.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("💾 HTML salvat în: debug_product_page.html (pentru debugging)\n")

        # Așteaptă puțin să poți vedea
        print("⏳ Browser-ul va rămâne deschis 10 secunde pentru verificare vizuală...")
        time.sleep(10)

    finally:
        driver.quit()
        print("\n✅ Test finalizat!")


if __name__ == "__main__":
    # PUNE AICI URL-ul unui produs de test
    test_url = input("Introdu URL-ul produsului eMAG (sau Enter pentru exemplu): ").strip()

    if not test_url:
        # URL de exemplu - înlocuiește cu unul real
        test_url = "https://www.emag.ro/telefon-mobil-samsung-galaxy-s24-ultra-5g-dual-sim-12gb-ram-256gb-titanium-gray-sm-s928bztgeue/pd/D23CPNMBM/"
        print(f"Folosesc URL exemplu: {test_url}\n")

    test_specifications(test_url)