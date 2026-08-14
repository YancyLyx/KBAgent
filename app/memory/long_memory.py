# -*- coding: utf-8 -*-
"""长期记忆模块

管理用户的长期记忆信息，使用 SQLite 持久化存储。
记录用户的偏好、历史话题等信息。
"""

import os
import sqlite3
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import yaml
import json


# 画像变更审计日志目录（人可读 MD，记录"变更前/变更后/来源/时间"）
PROFILES_AUDIT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "profiles",
)


class LongMemory:
    """长期记忆

    使用 SQLite 存储用户的长期信息，包括：
    - 用户基本信息
    - 历史查询话题
    - 频繁询问的主题
    """

    def __init__(self, config_path: str = "config/memory_config.yaml"):
        """
        初始化长期记忆

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        lt_config = self.config.get("long_term_memory", {})

        self.enabled = lt_config.get("enabled", True)
        self.db_path = lt_config.get("db_path", "./data/sqlite/memory.db")
        self.max_topics_per_user = lt_config.get("max_topics_per_user", 10)
        self.topic_expiry_days = lt_config.get("topic_expiry_days", 30)
        # 演进式摘要只保留最近 N 行（历史靠 Chroma 快照兜底），防止表无限增长
        self.summary_keep_recent = int(
            self.config.get("summary", {}).get("keep_recent", 20)
        )

        # 确保数据库目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # 初始化数据库
        self._init_database()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {
                "long_term_memory": {
                    "enabled": True,
                    "db_path": "./data/sqlite/memory.db",
                    "max_topics_per_user": 10,
                    "topic_expiry_days": 30
                }
            }

    def _init_database(self) -> None:
        """初始化数据库表结构"""
        if not self.enabled:
            return

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 用户记忆主表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    topic TEXT,
                    last_question TEXT,
                    question_count INTEGER DEFAULT 1,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    UNIQUE(user_id, topic)
                )
            """)

            # 用户画像表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    user_id TEXT PRIMARY KEY,
                    company TEXT,
                    contact_email TEXT,
                    preferences TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 对话摘要表（可选的高级功能）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    summary TEXT,
                    key_points TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    def update_memory(
        self,
        user_id: str,
        topic: Optional[str] = None,
        last_question: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        更新用户记忆

        Args:
            user_id: 用户 ID
            topic: 话题类型
            last_question: 最后提问内容
            metadata: 额外元数据

        Returns:
            是否成功
        """
        if not self.enabled or not user_id:
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 检查是否已存在该用户的该话题
                cursor.execute(
                    "SELECT id, question_count FROM user_memory WHERE user_id = ? AND topic = ?",
                    (user_id, topic)
                )
                result = cursor.fetchone()

                if result:
                    # 更新现有记录
                    memory_id, count = result
                    cursor.execute(
                        """
                        UPDATE user_memory
                        SET last_question = ?,
                            question_count = ?,
                            last_seen = CURRENT_TIMESTAMP,
                            metadata = ?
                        WHERE id = ?
                        """,
                        (
                            last_question or "",
                            count + 1,
                            json.dumps(metadata) if metadata else None,
                            memory_id
                        )
                    )
                else:
                    # 检查是否超过最大话题数
                    cursor.execute(
                        "SELECT COUNT(*) FROM user_memory WHERE user_id = ?",
                        (user_id,)
                    )
                    topic_count = cursor.fetchone()[0]

                    if topic_count >= self.max_topics_per_user:
                        # 删除最旧的话题
                        cursor.execute(
                            """
                            DELETE FROM user_memory
                            WHERE user_id = ? AND id = (
                                SELECT id FROM user_memory
                                WHERE user_id = ?
                                ORDER BY last_seen ASC
                                LIMIT 1
                            )
                            """,
                            (user_id, user_id)
                        )

                    # 插入新记录
                    cursor.execute(
                        """
                        INSERT INTO user_memory
                        (user_id, topic, last_question, metadata)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            topic or "general",
                            last_question or "",
                            json.dumps(metadata) if metadata else None
                        )
                    )

                conn.commit()
                return True

        except sqlite3.Error as e:
            print(f"更新长期记忆失败: {e}")
            return False

    def get_user_memory(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户记忆

        Args:
            user_id: 用户 ID

        Returns:
            用户记忆信息
        """
        if not self.enabled or not user_id:
            return {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # 获取用户所有话题
                cursor.execute(
                    """
                    SELECT topic, last_question, question_count,
                           first_seen, last_seen, metadata
                    FROM user_memory
                    WHERE user_id = ?
                    ORDER BY question_count DESC, last_seen DESC
                    """,
                    (user_id,)
                )

                topics = []
                for row in cursor.fetchall():
                    topics.append({
                        "topic": row["topic"],
                        "last_question": row["last_question"],
                        "question_count": row["question_count"],
                        "first_seen": row["first_seen"],
                        "last_seen": row["last_seen"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                    })

                # 获取用户画像
                cursor.execute(
                    "SELECT * FROM user_profile WHERE user_id = ?",
                    (user_id,)
                )
                profile_row = cursor.fetchone()

                profile = {}
                if profile_row:
                    profile = {
                        "company": profile_row["company"],
                        "contact_email": profile_row["contact_email"],
                        "preferences": json.loads(profile_row["preferences"]) if profile_row["preferences"] else {}
                    }

                return {
                    "user_id": user_id,
                    "topics": topics,
                    "profile": profile,
                    "total_interactions": sum(t["question_count"] for t in topics),
                    "frequent_topics": [t["topic"] for t in topics[:3]]
                }

        except sqlite3.Error as e:
            print(f"获取长期记忆失败: {e}")
            return {}

    def get_topics_by_type(self, user_id: str, topic: str) -> List[Dict[str, Any]]:
        """
        获取特定类型的历史话题

        Args:
            user_id: 用户 ID
            topic: 话题类型

        Returns:
            相关历史记录
        """
        if not self.enabled or not user_id:
            return []

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT last_question, question_count, last_seen
                    FROM user_memory
                    WHERE user_id = ? AND topic = ?
                    ORDER BY last_seen DESC
                    """,
                    (user_id, topic)
                )

                return [
                    {
                        "last_question": row["last_question"],
                        "question_count": row["question_count"],
                        "last_seen": row["last_seen"]
                    }
                    for row in cursor.fetchall()
                ]

        except sqlite3.Error as e:
            print(f"获取话题历史失败: {e}")
            return []

    def update_user_profile(
        self,
        user_id: str,
        company: Optional[str] = None,
        contact_email: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
        source: str = "manual",
    ) -> bool:
        """更新用户画像（按 key 合并，list 类型按文本去重 + 新时间戳覆盖）。

        写入规则（对应"偏好不一致"的处理）：
        - 同 key 同文本：保留 updated_at 更新的版本（乐观锁，防乱序覆盖）
        - 同 key 不同文本：合并并存（不同偏好可能对应不同场景，不盲目覆盖）
        - 每次更新后 append 一条人可读审计日志（data/profiles/{user_id}.md），
          记录变更前后与来源，供人工纠错/回溯
        """
        if not self.enabled or not user_id:
            return False

        try:
            old_row = self._get_profile_row(user_id)
            old_prefs = (old_row or {}).get("preferences", {})

            # 按 key 合并：list 类型做"文本去重 + 新时间戳覆盖"，其余整体替换
            merged_prefs = dict(old_prefs)
            for key, value in (preferences or {}).items():
                if isinstance(value, list):
                    merged_prefs[key] = self._merge_fact_list(
                        old_prefs.get(key, []), value
                    )
                else:
                    merged_prefs[key] = value

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO user_profile (user_id, company, contact_email, preferences)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        company = COALESCE(?, company),
                        contact_email = COALESCE(?, contact_email),
                        preferences = ?,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        user_id, company, contact_email,
                        json.dumps(merged_prefs, ensure_ascii=False),
                        company, contact_email,
                        json.dumps(merged_prefs, ensure_ascii=False),
                    )
                )

                conn.commit()

            # 人可读审计日志：变更前后 + 来源
            self._append_profile_audit(
                user_id,
                before={
                    "company": (old_row or {}).get("company"),
                    "contact_email": (old_row or {}).get("contact_email"),
                    "preferences": old_prefs,
                },
                after={
                    "company": company if company is not None else (old_row or {}).get("company"),
                    "contact_email": contact_email if contact_email is not None else (old_row or {}).get("contact_email"),
                    "preferences": merged_prefs,
                },
                source=source,
            )
            return True

        except sqlite3.Error as e:
            print(f"更新用户画像失败: {e}")
            return False

    # ------------------------------------------------------------------
    # 画像合并与审计（偏好冲突处理）
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_fact(fact) -> Optional[Dict]:
        """把画像条目归一化为 {text, updated_at, source}，兼容旧版纯字符串"""
        if isinstance(fact, str):
            text = fact.strip()
            if not text:
                return None
            return {"text": text, "updated_at": "", "source": ""}
        if isinstance(fact, dict):
            text = str(fact.get("text", "")).strip()
            if not text:
                return None
            return {
                "text": text,
                "updated_at": str(fact.get("updated_at", "")),
                "source": str(fact.get("source", "")),
            }
        return None

    def _merge_fact_list(self, old: list, new: list) -> list:
        """合并偏好列表：按文本去重，同文本保留 updated_at 更新的版本。

        新条目没有时间戳时用当前时间（LLM 提取的 key_facts 是纯字符串列表，
        在这里补上提取时间）；同文本但新时间戳更旧时丢弃（防乱序覆盖）。
        """
        now = datetime.now().isoformat()
        merged: Dict[str, Dict] = {}
        for f in old:
            nf = self._normalize_fact(f)
            if nf:
                merged[nf["text"]] = nf
        for f in new:
            nf = self._normalize_fact(f)
            if not nf:
                continue
            nf["updated_at"] = nf["updated_at"] or now
            nf["source"] = nf.get("source") or "extraction"
            old_f = merged.get(nf["text"])
            if not old_f or nf["updated_at"] >= old_f.get("updated_at", ""):
                merged[nf["text"]] = nf
        return list(merged.values())

    def _get_profile_row(self, user_id: str) -> Optional[Dict[str, Any]]:
        """读取 user_profile 行（供合并与审计用）"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT company, contact_email, preferences FROM user_profile WHERE user_id = ?",
                (user_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                "company": row["company"],
                "contact_email": row["contact_email"],
                "preferences": json.loads(row["preferences"]) if row["preferences"] else {},
            }

    def _append_profile_audit(
        self,
        user_id: str,
        before: Dict[str, Any],
        after: Dict[str, Any],
        source: str,
    ) -> None:
        """追加一条人可读画像变更记录（MD 格式，供审计/人工纠错）"""
        try:
            os.makedirs(PROFILES_AUDIT_DIR, exist_ok=True)
            path = os.path.join(PROFILES_AUDIT_DIR, f"{user_id}.md")
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"## {datetime.now().isoformat()}  source={source}\n")
                f.write(f"变更前：{json.dumps(before, ensure_ascii=False)}\n")
                f.write(f"变更后：{json.dumps(after, ensure_ascii=False)}\n\n")
        except Exception as e:
            print(f"写入画像审计日志失败: {e}")

    def add_summary(self, user_id: str, summary: str, key_points: str = "") -> bool:
        """保存对话摘要（自动清理：每个用户只保留最近 summary_keep_recent 行）"""
        if not self.enabled or not user_id:
            return False
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO conversation_summary (user_id, summary, key_points) VALUES (?, ?, ?)",
                    (user_id, summary, key_points)
                )
                cursor.execute(
                    """
                    DELETE FROM conversation_summary
                    WHERE user_id = ? AND id NOT IN (
                        SELECT id FROM conversation_summary
                        WHERE user_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    """,
                    (user_id, user_id, self.summary_keep_recent),
                )
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"保存摘要失败: {e}")
            return False

    def get_running_summary(self, user_id: str) -> str:
        """获取最新的对话摘要（演进式，只有一个）"""
        summaries = self.get_summaries(user_id, limit=1)
        return summaries[0]["summary"] if summaries else ""

    def get_summaries(self, user_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """获取用户最近的对话摘要"""
        if not self.enabled or not user_id:
            return []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT summary, key_points, created_at FROM conversation_summary WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit)
                )
                return [dict(r) for r in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"获取摘要失败: {e}")
            return []

    def cleanup_expired_memories(self) -> int:
        """
        清理过期的记忆

        Returns:
            清理的记录数
        """
        if not self.enabled:
            return 0

        expiry_date = datetime.now() - timedelta(days=self.topic_expiry_days)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "DELETE FROM user_memory WHERE last_seen < ?",
                    (expiry_date.isoformat(),)
                )

                deleted = cursor.rowcount
                conn.commit()
                return deleted

        except sqlite3.Error as e:
            print(f"清理过期记忆失败: {e}")
            return 0

    def clear_user_memory(self, user_id: str) -> bool:
        """清除指定用户的所有记忆（话题计数 + 画像 + 演进式摘要）"""
        if not self.enabled or not user_id:
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "DELETE FROM user_memory WHERE user_id = ?",
                    (user_id,)
                )
                cursor.execute(
                    "DELETE FROM user_profile WHERE user_id = ?",
                    (user_id,)
                )
                cursor.execute(
                    "DELETE FROM conversation_summary WHERE user_id = ?",
                    (user_id,)
                )

                conn.commit()
                return True

        except sqlite3.Error as e:
            print(f"清除用户记忆失败: {e}")
            return False

    def delete_profile_fact(self, user_id: str, fact_text: str) -> bool:
        """删除用户画像中的一条偏好（管理端人工纠错入口）。

        按文本精确匹配删除，删除前后都记入审计日志（source=admin_manual_edit），
        便于追溯"谁删了什么、为什么"。
        """
        if not self.enabled or not user_id or not fact_text:
            return False
        try:
            old_row = self._get_profile_row(user_id)
            if old_row is None:
                return False
            old_prefs = old_row.get("preferences", {})
            old_facts = list(old_prefs.get("extracted_facts", []))

            target = fact_text.strip()
            kept = []
            for f in old_facts:
                nf = self._normalize_fact(f)
                if nf is not None and nf["text"] != target:
                    kept.append(f)
            if len(kept) == len(old_facts):
                return False  # 没有匹配的条目

            new_prefs = dict(old_prefs)
            new_prefs["extracted_facts"] = kept
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE user_profile SET preferences = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                    (json.dumps(new_prefs, ensure_ascii=False), user_id),
                )
                conn.commit()

            self._append_profile_audit(
                user_id,
                before={"preferences": {"extracted_facts": old_facts}},
                after={"preferences": new_prefs},
                source="admin_manual_edit",
            )
            return True
        except sqlite3.Error as e:
            print(f"删除画像条目失败: {e}")
            return False
