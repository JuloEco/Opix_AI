# ============================================================================
# reasoning/prompts.py — Templates de prompts pour le mode Reasoning
#
# Reprend explicitement les 5 étapes du README (section 6) :
#   Compréhension → Décomposition → Résolution → Vérification → Réponse finale
#
# Les templates demandent un format structuré et facilement parsable, pour
# que reasoning/planner.py et reasoning/verifier.py puissent extraire chaque
# étape sans ambiguïté.
# ============================================================================

DECOMPOSITION_PROMPT = """Tu es Opsiom en mode raisonnement. Pour répondre à la question, tu dois \
d'abord la comprendre, puis la découper en étapes simples, avant de la résoudre.

Réponds STRICTEMENT dans ce format, une section par ligne :

COMPRÉHENSION: <reformule ce qui est demandé en une phrase>
ÉTAPES:
1. <première étape à résoudre>
2. <deuxième étape, si nécessaire>
3. <etc.>

Question : {question}
"""

RESOLUTION_PROMPT = """Tu es Opsiom en mode raisonnement. Voici une question et le plan d'étapes \
déjà établi. Résous chaque étape dans l'ordre, en montrant le calcul ou le \
raisonnement intermédiaire, puis donne la réponse finale.

Réponds STRICTEMENT dans ce format :

RÉSOLUTION:
1. <calcul ou raisonnement de l'étape 1> => <résultat intermédiaire>
2. <calcul ou raisonnement de l'étape 2> => <résultat intermédiaire>
RÉPONSE_FINALE: <réponse finale, complète et directement utilisable, sans les calculs>

Question : {question}

Plan :
{plan}
"""

VERIFICATION_PROMPT = """Tu es un vérificateur. Voici une question, le raisonnement suivi, et la \
réponse produite. Vérifie que chaque étape est correcte et que la réponse \
finale découle bien du raisonnement. Si tu détectes une erreur, corrige-la.

Réponds STRICTEMENT dans ce format :

VERDICT: <CORRECT ou INCORRECT>
CORRECTION: <réponse corrigée si INCORRECT, sinon recopie la réponse finale>

Question : {question}

Raisonnement :
{resolution}

Réponse proposée : {answer}
"""

DIRECT_ANSWER_PROMPT = """{question}"""
