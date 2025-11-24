#!/usr/bin/env python3
"""
Script de test rapide pour scraper une seule page du wiki Polytopia.
Utile pour tester avant de lancer un scraping complet.
"""

import sys
from pathlib import Path

# Importer le scraper principal
try:
    from scrape_wiki import PolytopiaWikiScraper
except ImportError:
    print("❌ Erreur: Impossible d'importer scrape_wiki.py")
    print("   Assurez-vous que scrape_wiki.py est dans le même dossier.")
    sys.exit(1)


def test_single_page():
    """Test le scraping d'une seule page"""
    
    print("🧪 Test du scraper Polytopia - Page unique\n")
    
    # URL de test : Map Generation (page bien structurée)
    test_url = "https://polytopia.fandom.com/wiki/Map_Generation"
    
    # Créer le scraper avec un dossier de test
    output_dir = Path("wiki_test")
    scraper = PolytopiaWikiScraper(
        output_dir=str(output_dir),
        delay=1.0  # Délai réduit pour le test
    )
    
    print(f"📁 Dossier de sortie: {output_dir.absolute()}\n")
    
    # Scraper la page
    success = scraper.scrape_page(test_url)
    
    if success:
        print("\n✅ Test réussi!")
        print(f"\n📂 Fichiers créés dans: {output_dir.absolute()}")
        
        # Lister les fichiers créés
        print("\n📄 Fichiers Markdown:")
        for md_file in output_dir.rglob("*.md"):
            rel_path = md_file.relative_to(output_dir)
            print(f"   - {rel_path}")
        
        print("\n🖼️  Images téléchargées:")
        img_dir = output_dir / "images"
        if img_dir.exists():
            images = list(img_dir.iterdir())
            if images:
                for img in images[:10]:  # Limiter l'affichage aux 10 premières
                    print(f"   - {img.name}")
                if len(images) > 10:
                    print(f"   ... et {len(images) - 10} autres images")
            else:
                print("   (aucune image)")
        
        print("\n💡 Pour voir le résultat:")
        print(f"   cat {output_dir}/game_mechanics/*.md")
        
    else:
        print("\n❌ Le test a échoué")
        print("   Vérifiez votre connexion Internet et les dépendances")
    
    return success


if __name__ == "__main__":
    print("=" * 60)
    print("⚠️  AVERTISSEMENT")
    print("=" * 60)
    print("Ce script teste le scraper sur UNE page du wiki Polytopia.")
    print("Le contenu est sous licence CC-BY-SA.")
    print("=" * 60)
    print("\nAppuyez sur Entrée pour continuer ou Ctrl+C pour annuler...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n❌ Annulé par l'utilisateur")
        sys.exit(0)
    
    print()
    
    try:
        success = test_single_page()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


