#!/usr/bin/env python3
"""
Script pour scraper le wiki Polytopia et créer une base de connaissance en markdown.
Respecte les bonnes pratiques de scraping (rate limiting, robots.txt).

Note: Le contenu du wiki Polytopia est sous licence CC-BY-SA.
Assurez-vous d'attribuer correctement la source.
"""

import os
import time
import re
import hashlib
import traceback
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote
from pathlib import Path
import html2text
from typing import Set, Dict, List
import argparse


class PolytopiaWikiScraper:
    def __init__(self, output_dir: str = "wiki_knowledge", delay: float = 2.0):
        """
        Initialise le scraper.
        
        Args:
            output_dir: Dossier de sortie pour les fichiers markdown
            delay: Délai en secondes entre chaque requête (rate limiting)
        """
        self.base_url = "https://polytopia.fandom.com"
        self.wiki_base = f"{self.base_url}/wiki/"
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.visited_urls: Set[str] = set()
        # Mapping URL -> chemin local du fichier (pour conversion des liens)
        self.url_to_filepath: Dict[str, str] = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PolytopiaKnowledgeBaseScraper/1.0 (Educational/Research Purpose)'
        })
        
        # Créer les dossiers de sortie
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        
        # Configuration html2text pour un meilleur rendu markdown
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.ignore_emphasis = False
        self.html_converter.body_width = 0  # Pas de wrap
        self.html_converter.protect_links = True
        
    def check_robots_txt(self):
        """Vérifie et affiche le contenu de robots.txt"""
        try:
            response = self.session.get(f"{self.base_url}/robots.txt")
            print("=" * 60)
            print("Contenu de robots.txt:")
            print("=" * 60)
            print(response.text)
            print("=" * 60)
            print("\n⚠️  Veuillez vérifier que le scraping est autorisé.")
            print("Appuyez sur Entrée pour continuer ou Ctrl+C pour annuler...")
            input()
        except Exception as e:
            print(f"Erreur lors de la lecture de robots.txt: {e}")
    
    def get_page_category(self, url: str) -> tuple[str, str]:
        """
        Détermine la catégorie et sous-catégorie d'une page pour l'organisation en dossiers.
        
        Returns:
            Tuple (category, subcategory) où subcategory peut être None
        """
        path = urlparse(url).path.lower()
        
        # Mapping des sous-catégories pour game_mechanics
        game_mechanics_subcategories = {
            'buildings': ['bridge', 'embassy', 'roads', 'temples', 'building'],
            'city': ['city', 'population', 'city_connection', 'city_upgrades'],
            'diplomacy': ['diplomacy', 'embassy', 'peace_treaty', 'tribe_relations', 'cloak', 'dagger', 'dinghy', 'pirate'],
            'technology': ['climbing', 'fishing', 'hunting', 'organization', 'riding', 'technology'],
            'abilities': ['burn_forest', 'clear_forest', 'destroy', 'grow_forest', 'starfish_harvesting', 'whale_hunting'],
        }
        
        # Vérifier les sous-catégories de game_mechanics d'abord
        for subcat, keywords in game_mechanics_subcategories.items():
            for keyword in keywords:
                if keyword in path:
                    return ('game_mechanics', subcat)
        
        # Catégories principales
        categories = {
            'tribes': ['xin-xi', 'imperius', 'bardur', 'oumaji', 'kickoo', 
                      'hoodrick', 'luxidoor', 'vengir', 'zebasi', 'ai-mo',
                      'quetzali', 'yădakk', 'aquarion', 'elyrion', 'polaris', 'cymanti'],
            'units': ['warrior', 'archer', 'defender', 'rider', 'mind_bender',
                     'swordsman', 'catapult', 'knight', 'giant', 'boat', 'ship',
                     'raft', 'rammer', 'scout', 'bomber', 'juggernaut'],
            'strategies': ['strategies', 'strategy', 'perfection', 'domination'],
            'game_mechanics': ['combat', 'movement', 'terrain', 'map_generation',
                              'score', 'stars', 'ruins', 'game_modes'],
        }
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in path:
                    return (category, None)
        
        return ('general', None)
    
    def extract_pages_from_category(self, category_url: str) -> List[str]:
        """
        Extrait toutes les pages membres d'une catégorie Fandom.
        
        Args:
            category_url: URL de la page de catégorie (ex: /wiki/Category:Strategies)
            
        Returns:
            Liste des URLs des pages membres de la catégorie
        """
        pages = []
        
        try:
            full_url = urljoin(self.base_url, category_url)
            print(f"\n📂 Extraction des pages depuis la catégorie: {full_url}")
            
            response = self.session.get(full_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Les pages de catégorie Fandom ont généralement une section avec les pages membres
            # Chercher dans les liens de la page
            content_div = soup.find('div', class_='mw-parser-output') or soup
            
            # Chercher dans les sections de catégorie Fandom
            # Les pages de catégorie ont souvent des sections avec des liens organisés
            category_sections = soup.find_all(['div', 'ul', 'table'], class_=re.compile(r'category|mw-category|allpages'))
            
            # Chercher aussi dans les tableaux de catégorie
            category_tables = soup.find_all('table', class_='wikitable')
            
            # Combiner toutes les sections pertinentes
            sections_to_check = list(category_sections) + list(category_tables) + [content_div]
            
            for section in sections_to_check:
                for link in section.find_all('a', href=True):
                    href = link['href']
                    
                    # Ignorer les liens de navigation, catégories, etc.
                    if (href.startswith('/wiki/') and 
                        not href.startswith('/wiki/Category:') and
                        not href.startswith('/wiki/File:') and
                        not href.startswith('/wiki/Template:') and
                        not href.startswith('/wiki/Special:') and
                        not href.startswith('/wiki/User:') and
                        not href.startswith('/wiki/Help:') and
                        ':' not in href.split('/wiki/')[-1]):
                        
                        # Nettoyer l'URL
                        clean_href = href.split('#')[0].split('?')[0]
                        full_page_url = urljoin(self.base_url, clean_href)
                        
                        if full_page_url not in pages:
                            pages.append(full_page_url)
            
            # Si aucune page trouvée, essayer une approche plus large
            if not pages:
                print("  ⚠️  Aucune page trouvée avec la méthode standard, tentative alternative...")
                # Chercher tous les liens dans la page principale
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if (href.startswith('/wiki/') and 
                        not any(href.startswith(f'/wiki/{prefix}:') for prefix in 
                               ['Category', 'File', 'Template', 'Special', 'User', 'Help']) and
                        ':' not in href.split('/wiki/')[-1]):
                        clean_href = href.split('#')[0].split('?')[0]
                        full_page_url = urljoin(self.base_url, clean_href)
                        if full_page_url not in pages:
                            pages.append(full_page_url)
            
            print(f"  ✓ {len(pages)} pages trouvées dans la catégorie")
            
            time.sleep(self.delay)
            
        except Exception as e:
            print(f"  ✗ Erreur lors de l'extraction de la catégorie: {e}")
            traceback.print_exc()
        
        return pages
    
    def sanitize_filename(self, title: str) -> str:
        """Nettoie un titre pour en faire un nom de fichier valide"""
        # Remplacer les caractères spéciaux
        filename = re.sub(r'[<>:"/\\|?*]', '_', title)
        filename = re.sub(r'\s+', '_', filename)
        filename = filename.strip('_')
        return filename[:200]  # Limiter la longueur
    
    def get_local_path_for_url(self, wiki_url: str) -> str:
        """
        Retourne le chemin relatif local pour une URL wiki.
        
        Args:
            wiki_url: URL complète du wiki (ex: https://polytopia.fandom.com/wiki/Warrior)
            
        Returns:
            Chemin relatif (ex: ../units/Warrior.md) ou None si la page n'a pas été scrapée
        """
        if wiki_url in self.url_to_filepath:
            return self.url_to_filepath[wiki_url]
        return None
    
    def convert_wiki_link_to_local(self, href: str, current_category: str) -> str:
        """
        Convertit un lien wiki (/wiki/PageName) en chemin relatif local.
        
        Args:
            href: Lien wiki (ex: /wiki/Warrior ou /wiki/Combat)
            current_category: Catégorie de la page actuelle (pour calculer le chemin relatif)
            
        Returns:
            Chemin relatif markdown (ex: ../units/Warrior.md) ou le href original si non trouvé
        """
        if not href.startswith('/wiki/'):
            return href
        
        # Construire l'URL complète
        full_url = urljoin(self.base_url, href)
        
        # Nettoyer l'URL (enlever les ancres et paramètres)
        if '#' in full_url:
            full_url = full_url.split('#')[0]
        if '?' in full_url:
            full_url = full_url.split('?')[0]
        
        # Vérifier si cette page a été scrapée
        local_path = self.get_local_path_for_url(full_url)
        if local_path:
            # Calculer le chemin relatif depuis la catégorie actuelle
            if current_category in local_path:
                # Même catégorie, juste le nom du fichier
                filename = os.path.basename(local_path)
                return filename
            else:
                # Catégorie différente, utiliser le chemin relatif
                return f"../{local_path}"
        
        # Si la page n'a pas encore été scrapée, retourner le href original
        # (elle sera scrapée plus tard)
        return href
    
    def get_extension_from_mime(self, content_type: str) -> str:
        """
        Convertit un type MIME en extension de fichier.
        
        Args:
            content_type: Type MIME (ex: 'image/png', 'image/jpeg')
            
        Returns:
            Extension avec le point (ex: '.png', '.jpg')
        """
        mime_to_ext = {
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg',
            'image/gif': '.gif',
            'image/webp': '.webp',
            'image/svg+xml': '.svg',
            'image/bmp': '.bmp',
            'image/x-icon': '.ico',
        }
        
        # Nettoyer le content-type (peut contenir des paramètres comme 'image/png; charset=utf-8')
        mime_type = content_type.split(';')[0].strip().lower()
        return mime_to_ext.get(mime_type, '.png')  # Par défaut .png si inconnu
    
    def download_image(self, img_url: str, alt_text: str = "") -> str:
        """
        Télécharge une image et retourne le chemin local.
        
        Args:
            img_url: URL de l'image
            alt_text: Texte alternatif pour nommer le fichier
            
        Returns:
            Chemin relatif de l'image téléchargée
        """
        try:
            # Construire l'URL complète
            full_url = urljoin(self.base_url, img_url)
            
            # Nettoyer l'URL (enlever les paramètres de requête pour le nom de fichier)
            parsed = urlparse(full_url)
            path_without_params = parsed.path
            
            # Générer un nom de fichier depuis le chemin
            original_name = os.path.basename(unquote(path_without_params))
            
            # Extraire l'extension depuis le nom original
            _, ext_from_url = os.path.splitext(original_name)
            
            # Télécharger l'image avec HEAD d'abord pour obtenir le Content-Type
            try:
                head_response = self.session.head(full_url, timeout=10, allow_redirects=True)
                content_type = head_response.headers.get('Content-Type', '')
                ext_from_mime = self.get_extension_from_mime(content_type) if content_type else ''
            except:
                ext_from_mime = ''
            
            # Utiliser l'extension de l'URL si présente, sinon celle du MIME, sinon .png par défaut
            extension = ext_from_url or ext_from_mime or '.png'
            
            # Générer le nom de fichier final
            if alt_text:
                # Utiliser le texte alternatif comme base
                base_name = self.sanitize_filename(alt_text)
                # Limiter la longueur pour éviter des noms trop longs
                if len(base_name) > 100:
                    base_name = base_name[:100]
                filename = base_name + extension
            else:
                # Utiliser le nom original ou générer un nom basé sur l'URL
                if original_name and original_name != '/':
                    base_name = self.sanitize_filename(original_name.replace(extension, ''))
                    if not base_name:
                        base_name = 'image'
                    filename = base_name + extension
                else:
                    # Générer un nom basé sur l'URL complète (hash)
                    url_hash = hashlib.md5(full_url.encode()).hexdigest()[:8]
                    filename = f"image_{url_hash}{extension}"
            
            # Chemin de destination
            img_path = self.output_dir / "images" / filename
            
            # Ne pas télécharger si déjà présent
            if img_path.exists():
                return f"images/{filename}"
            
            # Télécharger l'image complète
            response = self.session.get(full_url, timeout=10, allow_redirects=True)
            response.raise_for_status()
            
            # Vérifier que c'est bien une image
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                print(f"  ⚠️  L'URL ne semble pas être une image: {content_type}")
                # Continuer quand même, peut-être que c'est une image mal typée
            
            # Mettre à jour l'extension si nécessaire après le téléchargement
            if not ext_from_url and content_type:
                new_ext = self.get_extension_from_mime(content_type)
                if new_ext != extension:
                    # Renommer le fichier avec la bonne extension
                    old_filename = filename
                    filename = os.path.splitext(filename)[0] + new_ext
                    img_path = self.output_dir / "images" / filename
                    if (self.output_dir / "images" / old_filename).exists():
                        (self.output_dir / "images" / old_filename).unlink()
            
            # Sauvegarder l'image
            with open(img_path, 'wb') as f:
                f.write(response.content)
            
            print(f"  ✓ Image téléchargée: {filename} ({len(response.content)} bytes)")
            time.sleep(self.delay * 0.5)  # Délai réduit pour les images
            
            return f"images/{filename}"
            
        except Exception as e:
            print(f"  ✗ Erreur lors du téléchargement de l'image {img_url}: {e}")
            traceback.print_exc()
            return img_url  # Retourner l'URL originale en cas d'erreur
    
    def convert_table_to_markdown(self, table) -> str:
        """Convertit une table HTML en markdown"""
        rows = table.find_all('tr')
        if not rows:
            return ""
        
        markdown_table = []
        
        for i, row in enumerate(rows):
            cells = row.find_all(['th', 'td'])
            if not cells:
                continue
            
            # Extraire le texte de chaque cellule
            cell_texts = [cell.get_text(strip=True) for cell in cells]
            markdown_table.append("| " + " | ".join(cell_texts) + " |")
            
            # Ajouter la ligne de séparation après l'en-tête
            if i == 0:
                markdown_table.append("| " + " | ".join(["---"] * len(cells)) + " |")
        
        return "\n".join(markdown_table)
    
    def extract_content(self, soup: BeautifulSoup, current_category: str = 'general') -> Dict[str, any]:
        """
        Extrait le contenu principal d'une page wiki.
        
        Args:
            soup: BeautifulSoup object de la page
            current_category: Catégorie de la page actuelle (pour les liens relatifs)
        
        Returns:
            Dictionnaire avec title, content, images, tables
        """
        result = {
            'title': '',
            'content': '',
            'links': []
        }
        
        # Extraire le titre
        title_elem = soup.find('h1', class_='page-header__title')
        if title_elem:
            result['title'] = title_elem.get_text(strip=True)
        
        # Trouver le contenu principal
        content_div = soup.find('div', class_='mw-parser-output')
        if not content_div:
            return result
        
        # Supprimer les éléments indésirables
        for element in content_div.find_all(['script', 'style', 'nav']):
            element.decompose()
        
        # Traiter les images
        for img in content_div.find_all('img'):
            # Essayer plusieurs attributs pour trouver l'URL de l'image
            img_url = (img.get('src') or 
                      img.get('data-src') or 
                      img.get('data-image-key') or
                      img.get('data-lazy-src'))
            
            if img_url:
                # Nettoyer l'URL (enlever les paramètres de taille si présents)
                # Les URLs Fandom peuvent avoir des paramètres comme ?width=300
                if '?' in img_url:
                    img_url = img_url.split('?')[0]
                
                # Ignorer les images de décoration (sprites, icônes, etc.)
                if any(skip in img_url.lower() for skip in ['sprite', 'icon', 'logo', 'button']):
                    continue
                
                alt_text = img.get('alt', '') or img.get('title', '')
                
                # Télécharger l'image et obtenir le chemin local
                local_path = self.download_image(img_url, alt_text)
                
                # Mettre à jour la source dans le HTML
                img['src'] = local_path
        
        # Traiter les tableaux - les convertir en markdown
        for table in content_div.find_all('table'):
            if 'wikitable' in table.get('class', []):
                markdown_table = self.convert_table_to_markdown(table)
                # Remplacer le tableau par une balise temporaire
                table.replace_with(soup.new_string(f"\n\n{markdown_table}\n\n"))
        
        # Extraire et remplacer les liens internes vers d'autres pages wiki
        for link in content_div.find_all('a', href=True):
            href = link['href']
            if href.startswith('/wiki/'):
                # Nettoyer le href (enlever les ancres et paramètres)
                clean_href = href.split('#')[0].split('?')[0]
                
                # Ignorer les pages spéciales (Category:, File:, etc.)
                if ':' in clean_href and not clean_href.startswith('/wiki/File:'):
                    continue
                
                full_url = urljoin(self.base_url, clean_href)
                
                # Ajouter à la liste des liens à scraper
                if full_url not in self.visited_urls:
                    result['links'].append(full_url)
                
                # Remplacer le lien par un chemin local si disponible
                local_link = self.convert_wiki_link_to_local(clean_href, current_category)
                if local_link != clean_href:
                    link['href'] = local_link
        
        # Convertir en markdown
        html_content = str(content_div)
        markdown_content = self.html_converter.handle(html_content)
        
        # Note: Les liens seront mis à jour dans update_all_links() après le scraping complet
        # pour s'assurer que toutes les pages cibles ont été scrapées
        
        # Nettoyer le markdown
        markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)
        
        result['content'] = markdown_content
        
        return result
    
    def scrape_page(self, url: str) -> bool:
        """
        Scrape une page spécifique.
        
        Returns:
            True si succès, False sinon
        """
        if url in self.visited_urls:
            return False
        
        print(f"\n📄 Scraping: {url}")
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Déterminer la catégorie et sous-catégorie AVANT extract_content
            category, subcategory = self.get_page_category(url)
            
            # Extraire le contenu avec la catégorie
            content = self.extract_content(soup, current_category=category)
            
            if not content['title']:
                print("  ⚠️  Pas de titre trouvé, page ignorée")
                return False
            
            # Créer la structure de dossiers (catégorie/sous-catégorie)
            if subcategory:
                category_dir = self.output_dir / category / subcategory
                relative_path_base = f"{category}/{subcategory}"
            else:
                category_dir = self.output_dir / category
                relative_path_base = category
            
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Créer le fichier markdown
            filename = self.sanitize_filename(content['title']) + ".md"
            filepath = category_dir / filename
            relative_path = f"{relative_path_base}/{filename}"
            
            # Enregistrer le mapping URL -> fichier local
            self.url_to_filepath[url] = relative_path
            
            # Construire le contenu markdown avec métadonnées
            markdown_output = f"""# {content['title']}

**Source:** {url}  
**Licence:** CC-BY-SA  
**Date de scraping:** {time.strftime('%Y-%m-%d')}

---

{content['content']}

---

*Ce contenu est extrait du wiki Polytopia et est sous licence CC-BY-SA.*
"""
            
            # Écrire le fichier
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_output)
            
            print(f"  ✓ Sauvegardé: {relative_path}")
            print(f"  📎 {len(content['links'])} liens découverts")
            
            self.visited_urls.add(url)
            
            # Rate limiting
            time.sleep(self.delay)
            
            return True, content['links']
            
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
            traceback.print_exc()
            return False, []
    
    def scrape_wiki(self, start_urls: List[str], max_pages: int = None):
        """
        Scrape le wiki en partant des URLs de départ et en suivant récursivement tous les liens.
        
        Args:
            start_urls: Liste des URLs de départ
            max_pages: Nombre maximum de pages à scraper (None = illimité)
        """
        to_visit = list(start_urls)
        pages_scraped = 0
        
        print(f"\n🚀 Début du scraping du wiki Polytopia")
        if max_pages:
            print(f"   Nombre maximum de pages: {max_pages}")
        else:
            print(f"   Mode: Scraping complet (tous les liens)")
        print(f"   Délai entre requêtes: {self.delay}s")
        print(f"   Dossier de sortie: {self.output_dir.absolute()}")
        
        while to_visit:
            if max_pages and pages_scraped >= max_pages:
                print(f"\n⚠️  Limite de {max_pages} pages atteinte")
                break
            
            url = to_visit.pop(0)
            
            # Nettoyer l'URL (enlever ancres et paramètres)
            clean_url = url.split('#')[0].split('?')[0]
            
            # Ignorer les pages spéciales (Category:, Template:, etc.)
            if '/wiki/' in clean_url:
                page_name = clean_url.split('/wiki/')[-1]
                if ':' in page_name and not page_name.startswith('File:'):
                    continue  # Ignorer Category:, Template:, etc.
            
            success, discovered_links = self.scrape_page(clean_url)
            
            if success:
                pages_scraped += 1
                
                # Ajouter les nouveaux liens découverts à la file d'attente
                for link in discovered_links:
                    clean_link = link.split('#')[0].split('?')[0]
                    
                    # Ignorer les pages spéciales (Category:, File:, etc.)
                    if '/wiki/' in clean_link:
                        page_name = clean_link.split('/wiki/')[-1]
                        if ':' in page_name and not page_name.startswith('File:'):
                            continue  # Ignorer Category:, Template:, etc.
                    
                    # Ajouter si pas déjà visitée ou en file
                    if clean_link not in self.visited_urls and clean_link not in to_visit:
                        to_visit.append(clean_link)
                
                if max_pages:
                    print(f"   Progression: {pages_scraped}/{max_pages} pages | File: {len(to_visit)}")
                else:
                    print(f"   Progression: {pages_scraped} pages | File: {len(to_visit)}")
        
        print(f"\n✅ Scraping terminé!")
        print(f"   Pages scrapées: {pages_scraped}")
        print(f"   Dossier de sortie: {self.output_dir.absolute()}")
        
        # Mettre à jour tous les liens dans les fichiers markdown
        print(f"\n🔗 Mise à jour des liens internes...")
        self.update_all_links()
    
    def update_all_links(self):
        """
        Met à jour tous les liens wiki dans les fichiers markdown pour pointer vers les fichiers locaux.
        """
        updated_files = 0
        
        # Parcourir tous les fichiers markdown
        for md_file in self.output_dir.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Déterminer le chemin relatif du fichier actuel
                relative_path = md_file.relative_to(self.output_dir)
                current_parts = relative_path.parts[:-1]  # Exclure le nom du fichier
                
                original_content = content
                
                # Remplacer tous les liens /wiki/PageName par des chemins relatifs
                def replace_link_with_title(match):
                    """Remplace [text](</wiki/PageName> "title")"""
                    link_text = match.group(1)
                    wiki_path = match.group(2)
                    
                    # Nettoyer le chemin wiki
                    clean_path = wiki_path.split('#')[0].split('?')[0]
                    full_url = urljoin(self.base_url, f"/wiki/{clean_path}")
                    
                    # Trouver le chemin local
                    local_path = self.get_local_path_for_url(full_url)
                    if local_path:
                        # Calculer le chemin relatif
                        target_parts = Path(local_path).parts[:-1]  # Exclure le nom du fichier
                        filename = os.path.basename(local_path)
                        
                        # Si même catégorie et sous-catégorie
                        if current_parts == target_parts:
                            return f'[{link_text}]({filename})'
                        else:
                            # Calculer le nombre de remontées nécessaires
                            depth = len(current_parts)
                            if depth > 0:
                                return f'[{link_text}]({"../" * depth}{local_path})'
                            else:
                                return f'[{link_text}]({local_path})'
                    
                    return match.group(0)
                
                def replace_link_simple(match):
                    """Remplace [text](/wiki/PageName)"""
                    link_text = match.group(1)
                    wiki_path = match.group(2)
                    
                    # Nettoyer le chemin wiki
                    clean_path = wiki_path.split('#')[0].split('?')[0]
                    full_url = urljoin(self.base_url, f"/wiki/{clean_path}")
                    
                    # Trouver le chemin local
                    local_path = self.get_local_path_for_url(full_url)
                    if local_path:
                        # Calculer le chemin relatif
                        target_parts = Path(local_path).parts[:-1]  # Exclure le nom du fichier
                        filename = os.path.basename(local_path)
                        
                        # Si même catégorie et sous-catégorie
                        if current_parts == target_parts:
                            return f'[{link_text}]({filename})'
                        else:
                            # Calculer le nombre de remontées nécessaires
                            depth = len(current_parts)
                            if depth > 0:
                                return f'[{link_text}]({"../" * depth}{local_path})'
                            else:
                                return f'[{link_text}]({local_path})'
                    
                    return match.group(0)
                
                # Remplacer les liens au format [text](</wiki/PageName> "title")
                content = re.sub(
                    r'\[([^\]]+)\]\(</wiki/([^>]+)>[^)]*\)',
                    replace_link_with_title,
                    content
                )
                
                # Remplacer les liens au format [text](/wiki/PageName)
                content = re.sub(
                    r'\[([^\]]+)\]\(/wiki/([^\)]+)\)',
                    replace_link_simple,
                    content
                )
                
                # Sauvegarder si modifié
                if content != original_content:
                    with open(md_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    updated_files += 1
                    
            except Exception as e:
                print(f"  ⚠️  Erreur lors de la mise à jour de {md_file}: {e}")
        
        print(f"  ✓ {updated_files} fichiers mis à jour")
    
    def scrape_from_sitemap(self, max_pages: int = None):
        """
        Scrape le wiki en commençant par les catégories pertinentes et en suivant tous les liens.
        
        Args:
            max_pages: Nombre maximum de pages (None = illimité, scraping complet)
        """
        start_urls = []
        
        # Catégories à scraper
        categories_to_scrape = [
            "Category:Strategies",
            "Category:Game_Mechanics",
            "The_Battle_of_Polytopia_Wiki",  # Page d'accueil
        ]
        
        print(f"\n🎯 Extraction des pages depuis les catégories:")
        for category in categories_to_scrape:
            print(f"   - {category}")
        
        # Extraire les pages de chaque catégorie
        for category in categories_to_scrape:
            category_url = f"/wiki/{category}"
            
            if category.startswith("Category:"):
                # Extraire toutes les pages de la catégorie
                pages = self.extract_pages_from_category(category_url)
                start_urls.extend(pages)
            else:
                # Page normale, l'ajouter directement
                start_urls.append(f"{self.wiki_base}{category}")
        
        # Ajouter aussi quelques pages importantes pour s'assurer qu'elles sont incluses
        important_pages = [
            "Map_Generation",
            "Combat",
            "Movement",
            "Terrain",
        ]
        
        for page in important_pages:
            url = f"{self.wiki_base}{page}"
            if url not in start_urls:
                start_urls.append(url)
        
        print(f"\n📋 Total: {len(start_urls)} pages de départ")
        print(f"   Le script suivra automatiquement tous les liens internes découverts")
        
        self.scrape_wiki(start_urls, max_pages=max_pages)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape le wiki Polytopia pour créer une base de connaissance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python scrape_wiki.py --check-robots
  python scrape_wiki.py --max-pages 20
  python scrape_wiki.py --output ./knowledge --delay 3.0
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        default='wiki_knowledge',
        help='Dossier de sortie (défaut: wiki_knowledge)'
    )
    
    parser.add_argument(
        '--delay', '-d',
        type=float,
        default=2.0,
        help='Délai entre requêtes en secondes (défaut: 2.0)'
    )
    
    parser.add_argument(
        '--max-pages', '-m',
        type=int,
        default=None,
        help='Nombre maximum de pages à scraper (défaut: illimité, scraping complet)'
    )
    
    parser.add_argument(
        '--check-robots',
        action='store_true',
        help='Vérifier robots.txt avant de commencer'
    )
    
    args = parser.parse_args()
    
    scraper = PolytopiaWikiScraper(
        output_dir=args.output,
        delay=args.delay
    )
    
    if args.check_robots:
        scraper.check_robots_txt()
    
    scraper.scrape_from_sitemap(max_pages=args.max_pages)


if __name__ == "__main__":
    main()


