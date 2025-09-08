import sqlite3
from collections import deque
from typing import List, Optional
import openai
import requests  # Ajouté pour Gemini
import time

# --- Classe Memory avec SQLite et trim automatique ---
class Memory:
    """
    Classe Memory : gère la mémoire courte (volatile) et longue (persistante via SQLite) de l'agent conversationnel.
    - Mémoire courte : stocke les derniers messages pour le contexte immédiat (FIFO, trim par taille/token).
    - Mémoire longue : stocke des informations importantes de façon persistante (clé/valeur).
    Utilisation :
        memory = Memory(short_size=10, db_path="agent_memory.db")
        memory.add_short("message")
        memory.add_long("clé", "valeur")
    """
    def __init__(self, short_size: int = 10, db_path: str = "agent_memory.db", max_tokens_short: int = 2000):
        """
        Initialise la mémoire courte et longue.
        :param short_size: Nombre maximum de messages en mémoire courte
        :param db_path: Chemin du fichier SQLite pour la mémoire longue
        :param max_tokens_short: Limite de tokens pour la mémoire courte (trim automatique)
        """
        self.short_context = deque(maxlen=short_size)
        self.db_path = db_path
        self.max_tokens_short = max_tokens_short
        self._init_db()

    def _init_db(self):
        """
        Crée la table SQLite pour la mémoire longue si elle n'existe pas.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS long_memory (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.commit()
        conn.close()

    # Mémoire courte
    def add_short(self, message: str):
        """
        Ajoute un message à la mémoire courte et effectue un trim si besoin.
        :param message: Message à ajouter
        """
        self.short_context.append(message)
        self.trim_short()

    def get_short(self) -> List[str]:
        """
        Retourne la liste des messages en mémoire courte.
        :return: Liste de messages
        """
        return list(self.short_context)

    def reset_short(self):
        """
        Vide la mémoire courte.
        """
        self.short_context.clear()

    def trim_short(self):
        """
        Supprime les plus anciens messages si la limite de tokens est dépassée.
        """
        while sum(len(m.split()) for m in self.short_context) > self.max_tokens_short:
            self.short_context.popleft()

    # Mémoire longue
    def add_long(self, key: str, value: str):
        """
        Ajoute ou met à jour une entrée dans la mémoire longue (persistante).
        :param key: Clé unique
        :param value: Valeur à stocker
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO long_memory (key, value) VALUES (?, ?)", (key, value))
        conn.commit()
        conn.close()

    def get_long(self, key: str) -> Optional[str]:
        """
        Récupère une valeur de la mémoire longue à partir de sa clé.
        :param key: Clé recherchée
        :return: Valeur ou None
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT value FROM long_memory WHERE key = ?", (key,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None

    def get_all_long(self) -> List[str]:
        """
        Retourne toutes les valeurs stockées dans la mémoire longue.
        :return: Liste de valeurs
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT value FROM long_memory")
        rows = c.fetchall()
        conn.close()
        return [row[0] for row in rows]

# --- Classe Agent AI avancée ---
class Agent:
    """
    Classe Agent : agent conversationnel intelligent, multi-provider (OpenAI, Gemini).
    - Gère la mémoire (Memory)
    - Construit les prompts avec contexte
    - Sélectionne dynamiquement le modèle selon le provider
    - Route les requêtes vers OpenAI ou Gemini selon le provider
    Utilisation :
        agent = Agent(api_key, provider="openai" ou "gemini")
        response = agent.send_prompt(user_id, message, commands)
    """
    MODEL_PRIORITY = {
        "openai": [
            "gpt-5", "gpt-5-mini",
            "gpt-4.1", "gpt-4.1-mini",
            "gpt-4o", "gpt-4o-mini",
            "gpt-3.5-turbo"
        ],
        "gemini": [
            "gemini-1.5-flash", "gemini-1.5-pro"
        ]
    }
    DEFAULT_MODEL = {
        "openai": "gpt-4o",
        "gemini": "gemini-1.5-flash"
    }

    def __init__(self, default_model: str = None, memory_config: dict = None, provider: str = "openai"):
        """
        Initialise l'agent conversationnel.
        :param api_key: Clé API (OpenAI ou Gemini)
        :param default_model: Modèle à utiliser (optionnel, sinon choisi dynamiquement)
        :param memory_config: Configuration de la mémoire (dictionnaire)
        :param provider: Provider à utiliser ("openai" ou "gemini")
        """
        self.provider = provider
        self.api_key = self.get_api_key()
        self.model = default_model or self.DEFAULT_MODEL.get(provider, "gpt-4o")
        self.memory = Memory(**(memory_config or {}))
    
    def get_api_key(self, conf_path="api_conf.conf"):
        key_name = {
            "openai": "openai_api_key",
            "gemini": "gemini_api_key"
        }[self.provider]
        with open(conf_path, "r") as f:
            for line in f:
                if line.startswith(key_name):
                    return line.split("=", 1)[1].strip()
        raise ValueError(f"Clé API pour {self.provider} non trouvée dans le fichier de configuration.")

    def set_api_key(self, conf_path="api_conf.conf", new_key: str = None):  
        key_name = {
            self.provider: self.provider+"_api_key",
        }[self.provider]
        with open(conf_path, "a") as f:
            f.write(f"{key_name}={new_key}\n")
            self.api_key = new_key
            self.model = None

    def update_api_key(self, new_key: str):
        """
        Met à jour la clé API de l'agent.
        :param new_key: Nouvelle clé API
        """
        self.api_key = new_key

    def select_model(self, commands: Optional[List[str]] = None) -> str:
        """
        Sélectionne dynamiquement le modèle à utiliser selon le provider et les commandes.
        :param commands: Liste de commandes optionnelles
        :return: Nom du modèle
        """
        if self.provider == "gemini":
            return self.model or self.DEFAULT_MODEL["gemini"]
        if commands:
            if any(cmd in ["summarize", "translate"] for cmd in commands):
                return "gpt-4o-mini"
        return self.model or self.DEFAULT_MODEL["openai"]

    def build_prompt(self, user_id: str, message: str, commands: Optional[List[str]] = None) -> str:
        """
        Construit le prompt complet envoyé au LLM, incluant le contexte mémoire et les commandes.
        :param user_id: Identifiant utilisateur
        :param message: Message utilisateur
        :param commands: Liste de commandes optionnelles
        :return: Prompt complet (str)
        """
        system_prompt = "Vous êtes un assistant intelligent et utile."
        short_ctx = "\n".join(self.memory.get_short())
        long_ctx = "\n".join(self.memory.get_all_long())
        cmd_prompt = f"Commandes : {commands}" if commands else ""
        prompt = f"{system_prompt}\nMémoire courte:\n{short_ctx}\nMémoire longue:\n{long_ctx}\n{cmd_prompt}\nUtilisateur({user_id}): {message}"
        return prompt

    def post_process(self, response: str, commands: Optional[List[str]] = None) -> dict:
        """
        Post-traite la réponse brute du LLM pour ajouter des champs (résumé, sentiment, etc.).
        :param response: Réponse brute du LLM
        :param commands: Liste de commandes optionnelles
        :return: Dictionnaire enrichi
        """
        result = {"content": response}
        if commands:
            if "summarize" in commands:
                result["summary"] = response[:100] + "..."
            if "sentiment" in commands:
                result["sentiment"] = {"score": 0.0, "label": "neutral"}
        return result

    def send_prompt(self, user_id: str, message: str, commands: Optional[List[str]] = None) -> dict:
        """
        Envoie un prompt à l'API LLM (OpenAI ou Gemini) et gère la mémoire.
        :param user_id: Identifiant utilisateur
        :param message: Message utilisateur
        :param commands: Liste de commandes optionnelles
        :return: Dictionnaire réponse post-traitée
        """
        prompt = self.build_prompt(user_id, message, commands)
        selected_model = self.select_model(commands)

        if self.provider == "openai":
            openai.api_key = self.api_key
            try:
                resp = openai.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=500
                )
                response_text = resp.choices[0].message.content
            except Exception as e:
                response_text = f"Erreur lors de l'appel OpenAI : {str(e)}"
        elif self.provider == "gemini":
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent"
                headers = {"Content-Type": "application/json"}
                params = {"key": self.api_key}
                data = {
                    "contents": [
                        {"parts": [{"text": prompt}]}
                    ]
                }
                r = requests.post(url, headers=headers, params=params, json=data)
                r.raise_for_status()
                result = r.json()
                response_text = result["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                response_text = f"Erreur lors de l'appel Gemini : {str(e)}"
        else:
            response_text = "Provider non supporté."

        # Mise à jour mémoire
        self.memory.add_short(f"Utilisateur({user_id}): {message}")
        self.memory.add_short(f"Agent: {response_text}")
        self.memory.add_long(f"{user_id}_{len(self.memory.get_short())}", message)

        return self.post_process(response_text, commands)

# --- Classes de test ---
class TestClient:
    """
    TestClient permet de simuler un client pour tester les interactions avec la classe Agent.
    Utilisation typique :
        - Instancier avec la clé API, le provider ("openai" ou "gemini") et éventuellement une config mémoire.
        - Utiliser la méthode send pour envoyer un message et afficher la réponse.
    """
    def __init__(self, memory_config: dict = None, provider: str = "openai", default_model: str = None):
        """
        Initialise un TestClient.
        :param api_key: Clé API à utiliser (OpenAI ou Gemini)
        :param memory_config: Configuration de la mémoire (dictionnaire)
        :param provider: Provider à utiliser ("openai" ou "gemini")
        :param default_model: Modèle à utiliser (optionnel, sinon choisi dynamiquement)
        """
        self.agent = Agent(api_key, default_model=default_model, memory_config=memory_config, provider=provider)

    def send(self, user_id: str, message: str, commands: Optional[List[str]] = None):
        """
        Envoie un message à l'agent et affiche la réponse.
        :param user_id: Identifiant utilisateur
        :param message: Message à envoyer
        :param commands: Liste de commandes optionnelles (ex: ["summarize"])
        :return: Réponse de l'agent (dict)
        """
        response = self.agent.send_prompt(user_id, message, commands)
        print(f"[TestClient] Réponse: {response}")
        return response

class TestAgent:
    """
    TestAgent permet de réaliser des tests structurés et avancés sur la classe Agent.
    Il propose plusieurs scénarios de test :
        - test_message_simple : test d'un message basique
        - test_message_avec_commandes : test avec commande (ex: summarize)
        - test_memoire : test avancé de la mémoire (remplissage, trim, contexte)
    Utilisation typique :
        - Instancier avec la clé API, le provider ("openai" ou "gemini") et éventuellement une config mémoire.
        - Appeler run_all_tests() pour exécuter tous les scénarios.
    """
    def __init__(self, memory_config: dict = None, provider: str = "openai", default_model: str = None):
        """
        Initialise un TestAgent.
        :param api_key: Clé API à utiliser (OpenAI ou Gemini)
        :param memory_config: Configuration de la mémoire (dictionnaire)
        :param provider: Provider à utiliser ("openai" ou "gemini")
        :param default_model: Modèle à utiliser (optionnel, sinon choisi dynamiquement)
        """
        self.agent = Agent( default_model=default_model, memory_config=memory_config, provider=provider)

    def test_message_simple(self):
        """
        Teste l'envoi d'un message simple à l'agent et affiche la réponse.
        """
        user_id = "test_user"
        message = "Bonjour, que peux-tu faire ?"
        print("[TestAgent] Test message simple...")
        response = self.agent.send_prompt(user_id, message)
        print(f"[TestAgent] Réponse: {response}")
        return response

    def test_message_avec_commandes(self):
        """
        Teste l'envoi d'un message avec une commande (ex: summarize) et affiche la réponse.
        """
        user_id = "test_user"
        message = "Peux-tu résumer ce texte ?"
        commands = ["summarize"]
        print("[TestAgent] Test message avec commande 'summarize'...")
        response = self.agent.send_prompt(user_id, message, commands)
        print(f"[TestAgent] Réponse: {response}")
        return response

    def test_memoire(self):
        """
        Teste la gestion de la mémoire de l'agent :
            - Envoie plusieurs messages pour remplir la mémoire courte
            - Affiche la mémoire courte après chaque message
            - Affiche la mémoire longue
            - Vérifie l'utilisation du contexte dans une nouvelle requête
        """
        user_id = "memoire_user"
        print("[TestAgent] Test mémoire avancée...")
        messages = [
            "Premier message de test.",
            "Deuxième message pour la mémoire.",
            "Troisième message, on continue.",
            "Quatrième message, la mémoire courte va-t-elle suivre ?",
            "Cinquième message, testons le trim !"
        ]
        for i, msg in enumerate(messages):
            print(f"\n[Envoi {i+1}] {msg}")
            self.agent.send_prompt(user_id, msg)
            print(f"Mémoire courte après {i+1} messages : {self.agent.memory.get_short()}")
        print("\nMémoire longue :", self.agent.memory.get_all_long())
        # Vérification du contexte dans une nouvelle requête
        print("\n[Test du contexte dans une nouvelle requête]")
        response = self.agent.send_prompt(user_id, "Que sais-tu de notre conversation ?")
        print(f"Réponse avec contexte : {response}")
        return response

    def run_all_tests(self):
        """
        Exécute tous les scénarios de test disponibles sur l'agent.
        """
        self.test_message_simple()
        self.test_message_avec_commandes()
        self.test_memoire()
