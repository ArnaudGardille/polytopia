#!/usr/bin/env python3
"""Génère les icônes du manifest (icon-192.png et icon-512.png) à partir du logo existant."""

from pathlib import Path
from PIL import Image

def generate_manifest_icons():
    """Génère les icônes du manifest à partir du logo Site-logo.png."""
    project_root = Path(__file__).parent.parent
    logo_path = project_root / "frontend" / "public" / "icons" / "other" / "Site-logo.png"
    icons_dir = project_root / "frontend" / "public" / "icons"
    
    if not logo_path.exists():
        print(f"❌ Logo introuvable : {logo_path}")
        print("Création d'icônes placeholder simples...")
        # Créer des icônes placeholder simples
        for size in [192, 512]:
            icon = Image.new('RGB', (size, size), color='#2563eb')
            icon_path = icons_dir / f"icon-{size}.png"
            icon.save(icon_path)
            print(f"✅ Créé : {icon_path}")
        return
    
    print(f"📷 Utilisation du logo : {logo_path}")
    
    # Charger le logo
    logo = Image.open(logo_path)
    
    # Convertir en RGBA si nécessaire pour préserver la transparence
    if logo.mode != 'RGBA':
        logo = logo.convert('RGBA')
    
    # Générer les icônes aux tailles requises
    for size in [192, 512]:
        # Créer une nouvelle image avec fond blanc (pour le manifest)
        icon = Image.new('RGBA', (size, size), color=(255, 255, 255, 255))
        
        # Calculer les dimensions pour centrer le logo
        logo_width, logo_height = logo.size
        scale = min(size / logo_width, size / logo_height) * 0.9  # 90% pour laisser une marge
        new_width = int(logo_width * scale)
        new_height = int(logo_height * scale)
        
        # Redimensionner le logo
        resized_logo = logo.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Centrer le logo sur l'icône
        x_offset = (size - new_width) // 2
        y_offset = (size - new_height) // 2
        
        # Coller le logo sur l'icône
        icon.paste(resized_logo, (x_offset, y_offset), resized_logo)
        
        # Convertir en RGB pour le PNG final (sans transparence)
        icon_rgb = Image.new('RGB', icon.size, (255, 255, 255))
        icon_rgb.paste(icon, mask=icon.split()[3] if icon.mode == 'RGBA' else None)
        
        # Sauvegarder
        icon_path = icons_dir / f"icon-{size}.png"
        icon_rgb.save(icon_path, 'PNG')
        print(f"✅ Créé : {icon_path} ({size}x{size})")

if __name__ == "__main__":
    generate_manifest_icons()
    print("✨ Icônes du manifest générées avec succès !")


