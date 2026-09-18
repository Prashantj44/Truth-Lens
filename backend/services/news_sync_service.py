import asyncio
import csv
import logging
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.config import DATA_DIR
from backend.database.db import (
    save_news_article,
    is_news_url_indexed,
    get_recent_news_articles,
    get_news_stats,
    save_document
)
from backend.services.document_processor import document_processor
from backend.services.retrieval_service import retrieval_service

logger = logging.getLogger("truthlens.news_sync")

# Trusted Global News & Fact-Checking RSS Feeds
TRUSTED_RSS_FEEDS = [
    {
        "name": "Reuters World News",
        "url": "https://news.google.com/rss/search?q=when:24h+allinurl:reuters.com&ceid=US:en&hl=en-US&gl=US",
        "category": "WORLD_NEWS",
        "credibility": 98.0
    },
    {
        "name": "BBC News Top Stories",
        "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
        "category": "GLOBAL_AFFAIRS",
        "credibility": 96.0
    },
    {
        "name": "Associated Press News",
        "url": "https://news.google.com/rss/search?q=when:24h+allinurl:apnews.com&ceid=US:en&hl=en-US&gl=US",
        "category": "BREAKING_NEWS",
        "credibility": 97.0
    },
    {
        "name": "PolitiFact Verified Fact-Checks",
        "url": "https://www.politifact.com/rss/factchecks/",
        "category": "FACT_CHECK",
        "credibility": 99.0
    },
    {
        "name": "Google News International",
        "url": "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
        "category": "GENERAL_NEWS",
        "credibility": 90.0
    }
]

class NewsSyncService:
    def __init__(self):
        self.is_running = False
        self.last_sync_time: Optional[str] = None
        self.next_sync_time: Optional[str] = None
        self.sync_interval_hours = 24
        self.background_task: Optional[asyncio.Task] = None
        self.is_syncing = False

    def clean_html(self, raw_html: str) -> str:
        """Strip HTML tags, entities, and excessive whitespace."""
        if not raw_html:
            return ""
        clean = re.sub(r'<[^>]+>', ' ', raw_html)
        clean = re.sub(r'&[a-zA-Z0-9#]+;', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    def fetch_rss_feed(self, feed_cfg: Dict[str, Any], max_items: int = 15) -> List[Dict[str, Any]]:
        """
        Fetch and parse articles from a trusted RSS feed.
        """
        articles = []
        url = feed_cfg["url"]
        source_name = feed_cfg["name"]
        category = feed_cfg.get("category", "NEWS")

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "TruthLens-FactVerification/2.0 (Academic/Research RAG System)"}
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            items = root.findall(".//item")

            for item in items[:max_items]:
                title_elem = item.find("title")
                desc_elem = item.find("description")
                link_elem = item.find("link")
                pub_date_elem = item.find("pubDate")

                title = self.clean_html(title_elem.text if title_elem is not None else "")
                desc = self.clean_html(desc_elem.text if desc_elem is not None else "")
                link = link_elem.text.strip() if link_elem is not None and link_elem.text else ""
                pub_date = pub_date_elem.text.strip() if pub_date_elem is not None and pub_date_elem.text else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if not title:
                    continue

                full_content = f"HEADLINE: {title}\nSUMMARY: {desc}\nPUBLISHED: {pub_date}\nSOURCE: {source_name}"

                articles.append({
                    "title": title,
                    "description": desc,
                    "full_content": full_content,
                    "url": link,
                    "pub_date": pub_date,
                    "source_name": source_name,
                    "category": category,
                    "credibility": feed_cfg.get("credibility", 95.0)
                })

        except Exception as e:
            logger.warning(f"[NewsSyncService] Failed to fetch feed {source_name}: {e}")

        return articles

    def sync_live_news(self, max_per_feed: int = 10) -> Dict[str, Any]:
        """
        Execute an on-demand live news sync across all trusted feeds.
        Chunks articles and indexes them into ChromaDB vector database.
        """
        if self.is_syncing:
            return {"status": "in_progress", "message": "A news synchronization is already running."}

        self.is_syncing = True
        new_articles_count = 0
        total_chunks_indexed = 0
        ingested_items = []

        try:
            for feed in TRUSTED_RSS_FEEDS:
                articles = self.fetch_rss_feed(feed, max_items=max_per_feed)
                for art in articles:
                    url = art.get("url", "")
                    title = art["title"]

                    # Skip already indexed articles
                    if url and is_news_url_indexed(url):
                        continue

                    # Chunk the news content
                    content = art["full_content"]
                    chunks = document_processor.chunk_text(
                        text=content,
                        chunk_size=400,
                        overlap=60,
                        source_name=art["source_name"],
                        file_type="LIVE_RSS",
                        source_type="LIVE_NEWS_FEED",
                        extra_metadata={
                            "url": url,
                            "published_date": art["pub_date"],
                            "category": art["category"],
                            "credibility_score": art["credibility"]
                        }
                    )

                    if not chunks:
                        continue

                    # Index into ChromaDB
                    retrieval_service.add_chunks(chunks)

                    # Record in SQLite database
                    save_news_article(
                        title=title,
                        source_name=art["source_name"],
                        url=url,
                        content_snippet=art["description"][:300] if art["description"] else title,
                        published_date=art["pub_date"],
                        chunks_indexed=len(chunks),
                        category=art["category"]
                    )

                    new_articles_count += 1
                    total_chunks_indexed += len(chunks)
                    ingested_items.append({
                        "title": title,
                        "source": art["source_name"],
                        "chunks": len(chunks)
                    })

            self.last_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            next_time = datetime.now() + timedelta(hours=self.sync_interval_hours)
            self.next_sync_time = next_time.strftime("%Y-%m-%d %H:%M:%S")

            logger.info(f"[NewsSyncService] Synced {new_articles_count} new articles ({total_chunks_indexed} chunks).")

            return {
                "status": "success",
                "message": f"Successfully synced {new_articles_count} new live articles ({total_chunks_indexed} vector chunks).",
                "new_articles_count": new_articles_count,
                "total_chunks_indexed": total_chunks_indexed,
                "last_sync_time": self.last_sync_time,
                "next_sync_time": self.next_sync_time,
                "ingested_items": ingested_items[:15]
            }

        except Exception as e:
            logger.error(f"[NewsSyncService] Sync failed: {e}")
            return {"status": "error", "message": f"Sync failed: {str(e)}"}
        finally:
            self.is_syncing = False

    def train_on_existing_news_dataset(self, max_articles: int = 100) -> Dict[str, Any]:
        """
        Train / Index the RAG vector store on existing historical news datasets
        (ISOT Reuters True News & PolitiFact LIAR dataset).
        """
        indexed_count = 0
        total_chunks = 0

        # 1. Ingest from data/raw/isot/True.csv
        true_csv_path = DATA_DIR / "raw" / "isot" / "True.csv"
        if true_csv_path.exists():
            try:
                with open(true_csv_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for idx, row in enumerate(reader):
                        if idx >= max_articles:
                            break
                        
                        title = row.get("title", "").strip()
                        text = row.get("text", "").strip()
                        date_str = row.get("date", "").strip()
                        subject = row.get("subject", "politicsNews").strip()

                        if not title or not text or len(text) < 80:
                            continue

                        content = f"ARCHIVED REUTERS VERIFIED NEWS\nTITLE: {title}\nDATE: {date_str}\nSUBJECT: {subject}\n\n{text[:2000]}"
                        
                        chunks = document_processor.chunk_text(
                            text=content,
                            chunk_size=450,
                            overlap=80,
                            source_name="ISOT Reuters Verified Archive",
                            file_type="CSV_DATASET",
                            source_type="VERIFIED_NEWS_ARCHIVE",
                            extra_metadata={
                                "published_date": date_str,
                                "category": subject,
                                "credibility_score": 96.0
                            }
                        )

                        if chunks:
                            retrieval_service.add_chunks(chunks)
                            save_news_article(
                                title=title,
                                source_name="Reuters Verified Archive",
                                url=f"isot://true/{idx}",
                                content_snippet=text[:250],
                                published_date=date_str,
                                chunks_indexed=len(chunks),
                                category=subject
                            )
                            indexed_count += 1
                            total_chunks += len(chunks)
            except Exception as e:
                logger.warning(f"[NewsSyncService] Failed loading True.csv: {e}")

        # 2. Ingest from data/raw/liar/liar_all.csv
        liar_csv_path = DATA_DIR / "raw" / "liar" / "liar_all.csv"
        if liar_csv_path.exists():
            try:
                with open(liar_csv_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for idx, row in enumerate(reader):
                        if idx >= (max_articles // 2):
                            break
                        
                        statement = row.get("statement", "").strip()
                        label = row.get("label", "").strip()
                        speaker = row.get("speaker", "").strip()
                        subject = row.get("subject", "").strip()
                        context = row.get("context", "").strip()

                        if not statement:
                            continue

                        content = f"POLITIFACT FACT CHECK REPORT\nCLAIM: \"{statement}\"\nVERDICT: {label.upper()}\nSPEAKER: {speaker}\nCONTEXT: {context}\nSUBJECT: {subject}"
                        
                        chunks = document_processor.chunk_text(
                            text=content,
                            chunk_size=350,
                            overlap=50,
                            source_name="PolitiFact LIAR Ground Truth Archive",
                            file_type="CSV_DATASET",
                            source_type="FACT_CHECK_ARCHIVE",
                            extra_metadata={
                                "category": subject,
                                "speaker": speaker,
                                "credibility_score": 98.0
                            }
                        )

                        if chunks:
                            retrieval_service.add_chunks(chunks)
                            save_news_article(
                                title=f"Fact Check: {statement[:80]}...",
                                source_name="PolitiFact Archive",
                                url=f"liar://factcheck/{idx}",
                                content_snippet=f"Speaker: {speaker} | Truth Label: {label} | Statement: {statement}",
                                published_date="Historical",
                                chunks_indexed=len(chunks),
                                category=subject
                            )
                            indexed_count += 1
                            total_chunks += len(chunks)
            except Exception as e:
                logger.warning(f"[NewsSyncService] Failed loading liar_all.csv: {e}")

        save_document(
            filename="Historical News & Fact-Checking Dataset Training",
            source="ISOT Reuters & PolitiFact Dataset",
            source_type="VERIFIED_ARCHIVE",
            file_type="CSV",
            number_of_chunks=total_chunks
        )

        return {
            "status": "success",
            "message": f"Trained model on existing news dataset: Ingested {indexed_count} historical news articles & fact-checks ({total_chunks} vector chunks).",
            "articles_indexed": indexed_count,
            "total_chunks_indexed": total_chunks
        }

    async def start_daily_scheduler(self):
        """
        Background scheduler that runs every 24 hours to pull new updates.
        """
        self.is_running = True
        logger.info("[NewsSyncService] Background Daily News Scheduler started.")

        # Initial sync on startup
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.sync_live_news, 8)
        except Exception as e:
            logger.warning(f"[NewsSyncService] Initial startup sync error: {e}")

        while self.is_running:
            try:
                # Sleep for 24 hours (86400 seconds)
                await asyncio.sleep(self.sync_interval_hours * 3600)
                logger.info("[NewsSyncService] Executing scheduled daily news ingestion...")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.sync_live_news, 12)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[NewsSyncService] Error in daily scheduler loop: {e}")
                await asyncio.sleep(300)

    def get_status(self) -> Dict[str, Any]:
        """Return real-time scheduler health and stats."""
        stats = get_news_stats()
        recent = get_recent_news_articles(limit=20)
        
        return {
            "is_scheduler_running": self.is_running,
            "is_currently_syncing": self.is_syncing,
            "sync_interval_hours": self.sync_interval_hours,
            "last_sync_time": self.last_sync_time or stats.get("last_sync", "Never"),
            "next_sync_time": self.next_sync_time or "Within 24 hours",
            "total_live_articles": stats.get("total_articles", 0),
            "total_news_chunks": stats.get("total_news_chunks", 0),
            "active_feeds": [f["name"] for f in TRUSTED_RSS_FEEDS],
            "recent_articles": recent
        }

# Global Singleton Instance
news_sync_service = NewsSyncService()
