#!/usr/bin/env python3
"""Script pour télécharger toutes les images Polytopia depuis le wiki."""

import argparse
import re
import time
from pathlib import Path
from urllib.parse import urlparse, urljoin, unquote
import requests
from bs4 import BeautifulSoup


def get_wiki_pages_from_sitemap(base_url: str, max_pages: int = None) -> list[str]:
    """Récupère les URLs des pages principales du wiki."""
    print(f"Récupération des pages depuis {base_url}...")
    
    pages = [base_url]
    visited = {base_url}
    
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Trouver les liens vers les pages principales (catégories, unités, terrain, etc.)
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Filtrer les pages pertinentes du wiki
            if ('polytopia.fandom.com/wiki' in full_url and 
                full_url not in visited and
                not any(skip in full_url for skip in ['/User:', '/Special:', '/File:', '/Category:', '?action=', '#'])):
                
                # Limiter aux pages principales intéressantes
                interesting_keywords = [
                    'Unit', 'Terrain', 'Tile', 'City', 'Technology', 'Tribe',
                    'Warrior', 'Giant', 'Ship', 'Knight', 'Defender', 'Archer',
                    'Plain', 'Forest', 'Mountain', 'Water', 'Ocean'
                ]
                
                if any(kw.lower() in full_url.lower() for kw in interesting_keywords):
                    pages.append(full_url)
                    visited.add(full_url)
                    
                    if max_pages and len(pages) >= max_pages:
                        break
        
        print(f"  {len(pages)} pages trouvées")
        return pages
    
    except Exception as e:
        print(f"  Erreur: {e}")
        return pages


def extract_images_from_page(url: str) -> list[dict]:
    """Extrait toutes les images static.wikia.nocookie.net d'une page."""
    images = []
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Trouver toutes les images (img tags et autres éléments avec images)
        for img in soup.find_all('img'):
            # Essayer plusieurs attributs pour trouver l'URL
            src = (img.get('src') or 
                   img.get('data-src') or 
                   img.get('data-lazy-src') or
                   img.get('data-original'))
            
            if not src:
                continue
            
            # Filtrer les images de static.wikia.nocookie.net
            if 'static.wikia.nocookie.net' in src:
                # Nettoyer l'URL (enlever les paramètres de scale et revision)
                # Format: .../revision/latest/scale-to-width-down/250?cb=...
                if '/revision/' in src:
                    # Prendre l'URL avant /revision/
                    clean_url = src.split('/revision/')[0]
                else:
                    clean_url = src.split('?')[0]  # Enlever les query params
                
                # Extraire le nom du fichier
                parsed = urlparse(clean_url)
                filename = Path(unquote(parsed.path)).name
                
                # Ignorer les images trop petites ou non pertinentes
                if filename and not filename.startswith('.'):
                    images.append({
                        'url': clean_url,
                        'filename': filename,
                        'alt': img.get('alt', ''),
                        'title': img.get('title', ''),
                        'page_url': url,
                    })
        
        # Chercher aussi dans les divs et autres éléments avec des images en background
        for element in soup.find_all(attrs={'style': re.compile(r'static\.wikia\.nocookie\.net')}):
            style = element.get('style', '')
            urls = re.findall(r'url\(["\']?([^"\']*static\.wikia\.nocookie\.net[^"\']*)["\']?\)', style)
            for src in urls:
                if '/revision/' in src:
                    clean_url = src.split('/revision/')[0]
                else:
                    clean_url = src.split('?')[0]
                parsed = urlparse(clean_url)
                filename = Path(unquote(parsed.path)).name
                if filename and not filename.startswith('.'):
                    images.append({
                        'url': clean_url,
                        'filename': filename,
                        'alt': element.get('alt', ''),
                        'title': element.get('title', ''),
                        'page_url': url,
                    })
    
    except Exception as e:
        print(f"  Erreur lors de l'extraction des images de {url}: {e}")
    
    return images


def categorize_image(filename: str, alt: str, title: str, page_url: str) -> str:
    """Détermine la catégorie d'une image basée sur son nom et contexte."""
    filename_lower = filename.lower()
    alt_lower = alt.lower()
    title_lower = title.lower()
    page_lower = page_url.lower()
    
    # Catégories de terrain
    terrain_keywords = ['terrain', 'tile', 'plain', 'forest', 'mountain', 'water', 'ocean', 'land']
    if any(kw in filename_lower or kw in alt_lower or kw in title_lower for kw in terrain_keywords):
        return 'terrain'
    
    # Catégories d'unités
    unit_keywords = ['unit', 'warrior', 'giant', 'ship', 'knight', 'defender', 'archer', 'battleship']
    if any(kw in filename_lower or kw in alt_lower or kw in title_lower for kw in unit_keywords):
        return 'units'
    
    # Catégories de villes
    city_keywords = ['city', 'capital', 'village', 'town']
    if any(kw in filename_lower or kw in alt_lower or kw in title_lower for kw in city_keywords):
        return 'cities'
    
    # Catégories de technologies
    tech_keywords = ['tech', 'technology', 'research']
    if any(kw in filename_lower or kw in alt_lower or kw in title_lower for kw in tech_keywords):
        return 'tech'
    
    # Catégories de tribus
    tribe_keywords = ['tribe', 'civilization', 'faction']
    if any(kw in filename_lower or kw in alt_lower or kw in title_lower for kw in tribe_keywords):
        return 'tribes'
    
    # Par défaut, mettre dans "other"
    return 'other'


def download_image(url: str, output_path: Path) -> bool:
    """Télécharge une image depuis une URL."""
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"    Erreur lors du téléchargement de {url}: {e}")
        return False


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Télécharge toutes les images Polytopia depuis le wiki"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="frontend/public/icons",
        help="Dossier de sortie pour les images (défaut: frontend/public/icons)"
    )
    parser.add_argument(
        "--wiki-url",
        type=str,
        default="https://polytopia.fandom.com/wiki/The_Battle_of_Polytopia_Wiki",
        help="URL de base du wiki"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Nombre maximum de pages à parcourir (défaut: illimité)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Délai entre les requêtes en secondes (défaut: 0.5)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Téléchargement des images Polytopia depuis le wiki")
    print("=" * 60)
    
    # Récupérer les pages principales du wiki
    pages = get_wiki_pages_from_sitemap(args.wiki_url, args.max_pages)
    
    # Extraire toutes les images
    print(f"\nExtraction des images depuis {len(pages)} pages...")
    all_images = {}
    
    for i, page_url in enumerate(pages, 1):
        print(f"  [{i}/{len(pages)}] {page_url}")
        images = extract_images_from_page(page_url)
        
        for img in images:
            # Éviter les doublons (même URL)
            if img['url'] not in all_images:
                all_images[img['url']] = img
            else:
                # Merger les métadonnées si différentes
                existing = all_images[img['url']]
                if not existing['alt'] and img['alt']:
                    existing['alt'] = img['alt']
                if not existing['title'] and img['title']:
                    existing['title'] = img['title']
        
        time.sleep(args.delay)
    
    print(f"\n  Total: {len(all_images)} images uniques trouvées")
    
    # Télécharger les images
    print(f"\nTéléchargement des images...")
    downloaded = 0
    failed = 0
    
    for i, (url, img_info) in enumerate(all_images.items(), 1):
        category = categorize_image(
            img_info['filename'],
            img_info['alt'],
            img_info['title'],
            img_info['page_url']
        )
        
        # Créer le chemin de sortie
        category_dir = output_dir / category
        output_path = category_dir / img_info['filename']
        
        # Si le fichier existe déjà, on le skip
        if output_path.exists():
            print(f"  [{i}/{len(all_images)}] ✓ Déjà présent: {img_info['filename']}")
            downloaded += 1
            continue
        
        print(f"  [{i}/{len(all_images)}] Téléchargement: {img_info['filename']} -> {category}/")
        
        if download_image(url, output_path):
            downloaded += 1
        else:
            failed += 1
        
        time.sleep(args.delay)
    
    print("\n" + "=" * 60)
    print(f"Téléchargement terminé!")
    print(f"  ✓ Téléchargées: {downloaded}")
    print(f"  ✗ Échouées: {failed}")
    print(f"  📁 Dossier: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

