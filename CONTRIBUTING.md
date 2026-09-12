# Contribuer à ArgusAI

Merci de contribuer à ArgusAI. Le projet privilégie les améliorations simples, testables et compatibles avec une exécution locale.

## Avant de commencer

1. Ouvrir une issue pour les changements importants afin de confirmer le besoin.
2. Créer une branche depuis `main`.
3. Ne jamais ajouter de clé API, modèle téléchargé, document privé, trace d’exécution ou fichier `.env`.

## Préparer l’environnement

```bash
python -m venv .venv
# macOS / Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest -q
```

Pour la pile Docker, utiliser le profil Lite ou Full décrit dans le [README](README.md).

## Proposer une modification

1. Faire une modification ciblée et documentée.
2. Ajouter ou adapter un test lorsque le comportement change.
3. Lancer `pytest -q`.
4. Vérifier `git status` avant le commit.
5. Ouvrir une Pull Request qui explique le problème, la correction et les tests exécutés.

## Principes du projet

- Préserver l’exécution locale par défaut.
- Demander une confirmation avant tout accès web ou action externe.
- Préserver la compatibilité Windows et Docker.
- Préférer les dépendances optionnelles pour les fonctions lourdes ou expérimentales.
- Écrire des messages de commit clairs et orientés résultat.

## Signaler un problème de sécurité

Ne publiez pas de vulnérabilité ni de secret dans une issue. Contactez le mainteneur du dépôt de manière privée.
