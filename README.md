# 🧪 LLM Evaluation Platform

Une mini-plateforme pour **évaluer un système RAG** et **comparer deux versions de prompt** (V1 vs V2) sur 4 axes :
**Accuracy · Groundedness · Latency · Cost**.

> « Ship an evaluation suite with every feature » — ce projet montre exactement ça : pas juste un chatbot, mais la preuve mesurée qu'il fonctionne.

```
Question → Retrieval (BM25) → Prompt V1/V2 → LLM → Réponse
                                                    │
              Expected answer ──► Accuracy          │
              Retrieved docs  ──► Groundedness  ◄───┘
              Timer / tokens  ──► Latency · Cost
```

## ✅ Ce que contient le projet

| Élément | Détail |
|---|---|
| Documents | 5 fichiers de politique RH fictifs (`data/documents/`) |
| Jeu de test | **20 questions** (18 factuelles + 2 « sans réponse » pour détecter les hallucinations) |
| Prompts | `prompts/v1.json` (basique) vs `prompts/v2.json` (strict, cite les sources, s'abstient si incertain) |
| Interface | Streamlit : tableau V1 vs V2, graphiques, explorateur question par question, historique |
| CLI | `python -m llm_eval.cli run ...` + seuils pour la CI |
| Tests | 16 tests pytest + GitHub Action qui bloque le build si la qualité baisse |

## 🆓 Quelle IA gratuite utiliser ?

Tout passe par une API compatible OpenAI, donc tu peux changer de fournisseur sans toucher au code.

| Option | Prix | Pour qui | Mise en place |
|---|---|---|---|
| **Ollama** (local) | 100 % gratuit, illimité, offline | Si ton PC a 8 Go+ de RAM | Installer [ollama.com](https://ollama.com) puis `ollama pull llama3.2:3b` |
| **Groq** | Gratuit (quota) | Le plus simple et très rapide | Clé gratuite sur console.groq.com → `GROQ_API_KEY` |
| **Google Gemini** | Gratuit (quota) | Bonne qualité | Clé gratuite sur aistudio.google.com → `GEMINI_API_KEY` |
| **OpenRouter** | Modèles `:free` | Alternative | Clé sur openrouter.ai → `OPENROUTER_API_KEY` |
| **Mock** | Rien | Tester le pipeline sans aucune IA | `--provider mock` (aucune clé) |

> ⚠️ Les noms de modèles gratuits changent souvent. Si un modèle est retiré, change-le dans `.env` (`GROQ_MODEL=...`).
> Si tu as des erreurs `429` (quota), mets `REQUEST_DELAY=2` dans `.env`.

> 💡 **Recommandation** : Groq ou Ollama pour tes vrais résultats. Le mode `mock` sert seulement à vérifier que tout fonctionne : ses scores ne prouvent rien sur la qualité d'un vrai LLM. **Ne mets dans ton CV/portfolio que les chiffres d'un vrai modèle.**

## 🚀 Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # puis colle ta clé gratuite (Groq/Gemini)

# 1) Test sans IA (aucune clé)
python -m llm_eval.cli run --provider mock

# 2) Vraie IA gratuite
python -m llm_eval.cli run --provider groq --prompts v1 v2
python -m llm_eval.cli run --provider ollama --model llama3.2:3b

# 3) Interface web
streamlit run app.py

# 4) Tests
pytest -q
```

Exemple de sortie (format ; **les chiffres ci-dessous viennent du mode mock, pas d'un vrai LLM**) :

```
Metric           v1          v2          Delta
Accuracy         90%         95%         +5 pts
Groundedness     100%        100%        +0 pts
Retrieval hit@k  100%        100%        +0 pts
Avg latency      0.00s       0.00s       -0.00s
Cost / question  $0.00004    $0.00004    +0.00000$
```

## 📏 Comment les métriques sont calculées

- **Accuracy** (0/1 par question, moyenne) : tous les nombres de la réponse attendue doivent apparaître dans la réponse de l'IA + ≥ 60 % des mots-clés. Pour les questions sans réponse (`NOT_IN_DOCS`), l'IA doit répondre « je ne sais pas ».
- **Groundedness** : part des phrases de la réponse **soutenues par les documents récupérés** (≥ 70 % des mots présents dans le contexte et tous les chiffres présents). Une abstention vaut 1.0. ⚠️ Groundedness ≠ pertinence : une réponse fidèle au contexte mais hors-sujet est « grounded » mais fausse en accuracy — c'est pour ça qu'on mesure les deux.
- **Retrieval hit@k** : le bon document source est-il dans les `top_k` chunks ? (sépare les erreurs de retrieval des erreurs du LLM).
- **Latency** : temps retrieval + appel LLM (réussi), moyenne et P95.
- **Cost** : tokens réels renvoyés par l'API (sinon estimés) × prix de référence (`PRICE_INPUT_PER_1M` / `PRICE_OUTPUT_PER_1M`, par défaut gpt-4o-mini). Sur un provider gratuit, c'est un **coût équivalent** : « combien ça coûterait en production ».
- **Mode `--judge`** : le LLM note lui-même accuracy et groundedness (JSON). Plus nuancé que les heuristiques, mais plus lent et moins déterministe. Repli automatique sur les heuristiques si le JSON est invalide.

## 🧩 Étendre le projet

- **Nouvelle version de prompt** : crée `prompts/v3.json` (placeholders `{context}` et `{question}`), puis `--prompts v1 v2 v3`.
- **Nouvelles questions** : ajoute-les dans `data/test_set.json`.
- **Tes propres documents** : mets des `.md` (sections `## `) dans `data/documents/` et adapte le jeu de test.
- **Embeddings** : remplace `BM25Retriever` par un retriever vectoriel (même méthode `search(query, k)`).
- **Gate CI** : `--fail-under-accuracy 0.85` fait échouer le build en cas de régression.

## 🗂 Structure

```
app.py                  # interface Streamlit
llm_eval/
  config.py             # chemins, providers gratuits, prix
  retriever.py          # chargement + BM25
  llm.py                # client OpenAI-compatible (Ollama/Groq/Gemini/OpenRouter) + mock
  rag.py                # pipeline retrieve → prompt → generate
  metrics.py            # accuracy, groundedness, coût
  judge.py              # LLM-as-judge optionnel
  evaluator.py          # run, résumé, comparaison V1/V2, rapport
  cli.py
data/documents/  data/test_set.json  prompts/  tests/
```
