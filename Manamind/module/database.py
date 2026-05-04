import sqlite3
import json
import numpy as np
import faiss


class SkillDatabase:
    def __init__(self, database_path, json_path, embedding_model):
        self.json_path = json_path
        self.connection = sqlite3.connect(database_path)
        self.embedding = embedding_model
        self.create_tables()
        self.add_missing_columns()
        self.load_from_source(json_path)

    def create_tables(self):
        cursor = self.connection.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS categories (name TEXT PRIMARY KEY)")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT PRIMARY KEY,
                name TEXT,
                category TEXT,
                description TEXT,
                success_rate REAL
            )
        """)
        self.connection.commit()

    def add_missing_columns(self):
        cursor = self.connection.cursor()
        cursor.execute("PRAGMA table_info(skills)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        if "embedding" not in existing_columns:
            cursor.execute("ALTER TABLE skills ADD COLUMN embedding BLOB")
            self.connection.commit()

    def load_from_source(self, json_path):
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM skills")
        already_loaded = cursor.fetchone()[0] > 0
        if already_loaded:
            self.load_from_database()
            return

        with open(json_path, "r") as file:
            raw_data = json.load(file)

        all_skills = raw_data.get("skills", [])
        self.build_category_index(all_skills, cursor)
        self.build_skill_index(all_skills, cursor)
        self.connection.commit()

    def build_category_index(self, all_skills, cursor):
        unique_categories = list(set(
            skill.get("category") for skill in all_skills if skill.get("category")
        ))

        if not unique_categories:
            return
        
        skills_per_category = [[] for _ in unique_categories]
        for skill in all_skills:
            category = skill.get("category")
            if category in unique_categories:
                index = unique_categories.index(category)
                skills_per_category[index].append(skill)
                
        unique_categories_descriptions = []
        for category, skills in zip(unique_categories, skills_per_category):
            description = f"Category: {category}\n"
            for skill in skills:
                description += f"- {skill.get('name', '')}: {skill.get('description', '')}\n"
            unique_categories_descriptions.append(description)
            
        category_vectors = self.embedding.encode_document(unique_categories_descriptions)
        vector_dimension = len(category_vectors[0])

        self.category_index = faiss.IndexFlatIP(vector_dimension)
        self.category_names = []

        for category_name, vector in zip(unique_categories, category_vectors):
            normalized = vector / np.linalg.norm(vector)
            self.category_index.add(normalized.reshape(1, -1))
            self.category_names.append(category_name)
            cursor.execute("INSERT OR IGNORE INTO categories VALUES (?)", (category_name,))

    def build_skill_index(self, all_skills, cursor):
        self.skill_vectors = []
        self.skill_ids = []
        self.skill_data = {}

        for skill in all_skills:
            description = skill.get("description", "")
            skill_id = skill.get("skill_id", "")

            if not description:
                continue

            vector = self.embedding.encode_document([description])[0]
            normalized = vector / np.linalg.norm(vector)

            self.skill_vectors.append(normalized)
            self.skill_ids.append(skill_id)
            self.skill_data[skill_id] = {
                "skill_id": skill_id,
                "name": skill.get("name", ""),
                "category": skill.get("category", ""),
                "description": description,
                "success_rate": skill.get("success_rate", 0.0)
            }

            cursor.execute(
                "INSERT OR REPLACE INTO skills (skill_id, name, category, description, success_rate, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    skill_id,
                    self.skill_data[skill_id]["name"],
                    self.skill_data[skill_id]["category"],
                    description,
                    self.skill_data[skill_id]["success_rate"],
                    normalized.astype(np.float32).tobytes()
                )
            )

        self.skill_vectors = np.array(self.skill_vectors)

    def load_from_database(self):
        cursor = self.connection.cursor()

        cursor.execute("PRAGMA table_info(skills)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        has_embeddings = "embedding" in existing_columns

        if has_embeddings:
            cursor.execute("SELECT skill_id, name, category, description, success_rate, embedding FROM skills")
        else:
            cursor.execute("SELECT skill_id, name, category, description, success_rate FROM skills")

        skill_rows = cursor.fetchall()
        skill_vectors, skill_ids, skill_data = [], [], {}

        for row in skill_rows:
            if has_embeddings:
                skill_id, name, category, description, success_rate, embedding_blob = row
                vector = np.frombuffer(embedding_blob, dtype=np.float32) if embedding_blob else self.embedding.encode_document([description])[0]
            else:
                skill_id, name, category, description, success_rate = row
                vector = self.embedding.encode_document([description])[0]

            normalized = vector / np.linalg.norm(vector)
            skill_vectors.append(normalized)
            skill_ids.append(skill_id)
            skill_data[skill_id] = {
                "skill_id": skill_id,
                "name": name,
                "category": category,
                "description": description,
                "success_rate": success_rate
            }

        self.skill_vectors = np.array(skill_vectors)
        self.skill_ids = skill_ids
        self.skill_data = skill_data

        cursor.execute("SELECT name FROM categories")
        category_rows = cursor.fetchall()

        if category_rows:
            category_names = [row[0] for row in category_rows]
            category_vectors = self.embedding.encode_document(category_names)
            vector_dimension = len(category_vectors[0])

            self.category_index = faiss.IndexFlatIP(vector_dimension)
            self.category_names = []

            for category_name, vector in zip(category_names, category_vectors):
                normalized = vector / np.linalg.norm(vector)
                self.category_index.add(normalized.reshape(1, -1))
                self.category_names.append(category_name)

    def search(self, query, num_categories=3, num_results=5):
        query_vector = self.embedding.encode_query([query])[0]
        query_vector = query_vector / np.linalg.norm(query_vector)

        if hasattr(self, "category_index") and self.category_index.ntotal > 0:
            _, nearest_indices = self.category_index.search(query_vector.reshape(1, -1), num_categories)
            matched_categories = [self.category_names[i] for i in nearest_indices[0] if i != -1]
        else:
            matched_categories = []

        cursor = self.connection.cursor()

        if matched_categories:
            placeholders = ",".join(["?"] * len(matched_categories))
            cursor.execute(f"SELECT skill_id FROM skills WHERE category IN ({placeholders})", matched_categories)
            candidate_ids = {row[0] for row in cursor.fetchall()}
        else:
            candidate_ids = set(self.skill_ids)

        if not candidate_ids:
            return []

        candidate_positions = [i for i, sid in enumerate(self.skill_ids) if sid in candidate_ids]
        if not candidate_positions:
            return []

        candidate_vectors = self.skill_vectors[candidate_positions]
        similarity_scores = np.dot(candidate_vectors, query_vector)

        top_positions = np.argsort(similarity_scores)[::-1][:num_results]

        results = []
        for position in top_positions:
            global_position = candidate_positions[position]
            skill_id = self.skill_ids[global_position]
            results.append((skill_id, float(similarity_scores[position]), self.skill_data[skill_id]))

        return results

    def close(self):
        self.connection.close()