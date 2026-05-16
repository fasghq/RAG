import os
import json


class Agents:
    """
    Manages agents, storing them in a dictionary and optionally persisting it to a JSON file.
    """
    def __init__(self, db=None, local_path="agents.json"):
        """
        Initializes the Subscriptions object.

        Args:
            db (dict, optional): Initial database (dictionary) to use. Defaults to an empty dictionary.
            local_path (str, optional): Path to the JSON file for persistence. Defaults to "agents.json".
        """
        self.local_path = local_path
        if db is None:
            self.db = {}
        else:
            self.db = db


    @classmethod
    def from_json(cls, local_path="agents.json"):
        """
        Creates a Agents object from a JSON file.

        Args:
            local_path (str, optional): Path to the JSON file. Defaults to "agents.json".

        Returns:
            Agents: A new Agents object initialized with data from the JSON file, or an empty Agents object if the file doesn't exist or is empty/invalid.
        """
        if os.path.exists(local_path):
            try:
                with open(local_path, "r", encoding="utf-8") as file:
                    db = json.load(file)
                    return cls(db, local_path)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Error loading JSON from {local_path}: {e}.  Returning an empty Agents object.")
                return cls(db={}, local_path=local_path)
        else:
            print(f"File {local_path} not found. Returning an empty Agents object.")
            return cls(db={}, local_path=local_path)
        

    def to_json(self):
        """
        Saves Agents data to a JSON file.
        """
        if self.local_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.local_path)), exist_ok=True)
            try:
                with open(self.local_path, "w", encoding="utf-8") as file:
                    json.dump(self.db, file, ensure_ascii=False, indent=4)
            except (OSError, TypeError) as e:
                print(f"Error saving to JSON file {self.local_path}: {e}")
        else:
            print("Warning: local_path is empty.  Data not saved to JSON.")

    
    def add_agent(self, agent_id: str) -> bool:
        """
        Adds given agent to DB.

        Args:
            agent_id: The ID of agent to add.

        Returns:
            bool: True if new agent is registered, False otherwise, e.g. if agent_id already exists.
        """
        if agent_id not in self.db.keys():
            self.db[agent_id] = {"files": "", "documents": "", "chunks": "", "vectors": ""}
            return True
        return False


    def remove_agent(self, agent_id: str) -> bool:
        """
        Removes given agent from DB.

        Args:
            agent_id: The ID of agent to remove.

        Returns:
            bool: True if new agent is removed, False otherwise, e.g. if agent_id does not exist.
        """
        if agent_id in self.db.keys():
            del self.db[agent_id]
            return True
        return False
    
    
    def get_agent_field(self, agent_id: str, key: str) -> str:
        """
        Retrieves a value for a given agent key.

        Args:
            agent_id: The agent's ID.
            key: The key of the value to retrieve.

        Returns:
            The agent's value if it exists, an empty string "" otherwise.
        """
        value = ""
        try:
            value = self.db[agent_id][key]
        except Exception as e:
            print(str(e))
        return value
    

    def set_agent_field(self, agent_id: str, key: str, value: str) -> bool:
        """
        Sets a value for a given agent.

        Args:
            agent_id: The agent's ID.
            key: The key of value to set.
            value: The value to set.

        Returns:
            bool: True if operation succeeded, False otherwise, e.g. if agent_id does not exist.
        """
        if agent_id in self.db.keys():
            self.db[agent_id][key] = value
            return True
        return False
