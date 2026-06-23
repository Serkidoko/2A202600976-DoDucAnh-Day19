from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import re
import sys
import textwrap
import time
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
REPORT_DIR = ROOT / "reports"
LLM_CONTEXT_CHARS = 5000
LLM_JUDGE_ANSWER_CHARS = 2500
LLM_DISABLED = False


ENTITY_ALIASES: dict[str, list[str]] = {
    "United States": ["United States", "US", "U.S.", "America", "American", "Americans", "hoa ky", "my"],
    "China": ["China", "Chinese", "trung quoc"],
    "Europe": ["Europe", "European"],
    "California": ["California"],
    "New York": ["New York"],
    "Texas": ["Texas"],
    "Tesla": ["Tesla", "Model Y", "Model 3", "Supercharger", "Tesla Supercharger"],
    "Ford": ["Ford", "Mustang Mach-E", "F-150 Lightning"],
    "General Motors": ["General Motors", "GM", "Chevy Bolt", "Bolt"],
    "Hyundai": ["Hyundai", "Ioniq5", "Ioniq 5"],
    "Kia": ["Kia"],
    "Volvo": ["Volvo"],
    "BMW": ["BMW"],
    "Mercedes-Benz": ["Mercedes-Benz", "Mercedes", "Mercedes Benz"],
    "Audi": ["Audi"],
    "Volkswagen": ["Volkswagen", "VW"],
    "Rivian": ["Rivian"],
    "VinFast": ["VinFast", "Vinfast"],
    "ChargePoint": ["ChargePoint"],
    "Electrify America": ["Electrify America"],
    "EVgo": ["EVgo", "EV go"],
    "Blink": ["Blink"],
    "Cox Automotive": ["Cox Automotive", "Kelley Blue Book", "KBB"],
    "ICCT": ["ICCT", "International Council on Clean Transportation"],
    "BloombergNEF": ["BloombergNEF", "BNEF", "Bloomberg NEF"],
    "Goldman Sachs": ["Goldman Sachs"],
    "Deloitte": ["Deloitte"],
    "Pew Research Center": ["Pew Research Center", "Pew"],
    "Consumer Reports": ["Consumer Reports"],
    "McKinsey": ["McKinsey"],
    "J.D. Power": ["J.D. Power", "JD Power"],
    "Inflation Reduction Act": ["Inflation Reduction Act", "IRA"],
    "Infrastructure Investment and Jobs Act": ["Infrastructure Investment and Jobs Act"],
    "EV sales": [
        "EV sales",
        "electric vehicle sales",
        "sales growth",
        "market share",
        "adoption",
        "uptake",
        "doanh so",
    ],
    "EV slowdown": [
        "slowdown",
        "sales slowing",
        "slowing demand",
        "softening demand",
        "demand slowdown",
        "tang truong cham",
    ],
    "charging infrastructure": [
        "charging infrastructure",
        "public chargers",
        "charging stations",
        "charge points",
        "charging ports",
        "EVSE",
        "DC fast",
        "Level 2",
        "range anxiety",
        "ha tang sac",
        "tram sac",
        "sac",
    ],
    "consumer sentiment": [
        "consumer sentiment",
        "consumer attitudes",
        "consumer interest",
        "consumer confidence",
        "views",
        "confidence",
        "consider purchasing",
        "tam ly nguoi dung",
        "nguoi tieu dung",
    ],
    "investor sentiment": [
        "investor sentiment",
        "investment opportunities",
        "investors",
        "financial sentiment",
        "tam ly nha dau tu",
    ],
    "policy": [
        "policy",
        "government",
        "standards",
        "regulations",
        "chinh sach",
    ],
    "ZEV regulations": [
        "ZEV",
        "ZEV regulations",
        "zero emission vehicle regulations",
        "zero-emission vehicle regulations",
        "zero-emission vehicle",
    ],
    "consumer incentives": [
        "consumer incentives",
        "purchase incentives",
        "tax credits",
        "incentives",
        "fee reduction",
        "uu dai",
        "tin dung thue",
    ],
    "EV prices": [
        "prices",
        "average transaction price",
        "price cuts",
        "affordability",
        "used EV prices",
        "gia",
    ],
    "hybrids": ["hybrids", "HEV", "PHEV", "plug-in hybrids", "hybrid"],
    "automakers": ["automakers", "OEM", "manufacturers", "car manufacturers"],
    "battery": ["battery", "batteries", "LFP", "lithium iron phosphate"],
    "public health": ["public health", "clean air", "climate pollution", "CO2", "emissions"],
}


ENTITY_WEIGHTS = {
    "United States": 0.2,
    "EV sales": 0.45,
    "policy": 0.55,
    "automakers": 0.45,
}


EVALUATION_CASES = [
    {
        "question": "Is US EV demand really slowing, and which evidence supports both the slowdown and growth narratives?",
        "expected": [
            "Q1 2024 EV sales slowed or fell quarter over quarter",
            "2023 US EV sales passed 1 million",
            "ICCT or BNEF argue the slowdown narrative is exaggerated",
            "growth rate and consumer/infrastructure concerns still matter",
        ],
        "keywords": ["q1 2024", "1 million", "slowdown", "icct", "bnef", "growth"],
    },
    {
        "question": "How do charging infrastructure and consumer confidence affect EV adoption in the United States?",
        "expected": [
            "Charging availability and range anxiety are adoption barriers",
            "Pew reports limited confidence in future charging infrastructure",
            "large charger investments and 2030 targets are linked to adoption",
        ],
        "keywords": ["charging", "confidence", "range anxiety", "2030", "public chargers", "pew"],
    },
    {
        "question": "What role do Tesla price cuts and market share changes play in US EV sentiment?",
        "expected": [
            "Tesla cut prices and average transaction prices fell",
            "Tesla market share declined even while it remained the leader",
            "lower prices did not automatically create higher volume in Q1 2024",
        ],
        "keywords": ["tesla", "price", "market share", "q1", "lower", "volume"],
    },
    {
        "question": "How do policy incentives and ZEV regulations relate to regional EV uptake?",
        "expected": [
            "ZEV regulation states had higher EV shares and more models",
            "top metropolitan markets had many promotion actions and consumer incentives",
            "California and other states outperformed national averages",
        ],
        "keywords": ["zev", "incentives", "california", "metropolitan", "states", "models"],
    },
    {
        "question": "How does competition from China and hybrids change the outlook for US automakers and investors?",
        "expected": [
            "China has cost and supply-chain advantages in EVs and batteries",
            "hybrids are gaining attention during the EV slowdown",
            "investors may prefer automakers with strong balance sheets and multiple powertrains",
        ],
        "keywords": ["china", "hybrid", "balance sheet", "powertrains", "investors", "batteries"],
    },
    {
        "question": "Why can Q1 2024 look weak while the US EV market still has a positive 2024 outlook?",
        "expected": [
            "Q1 2024 sales fell quarter over quarter",
            "Cox Automotive still forecast full-year EV growth",
            "more products, incentives, leasing, inventory, and infrastructure can support sales",
        ],
        "keywords": ["q1 2024", "cox", "forecast", "incentives", "leasing", "infrastructure"],
    },
    {
        "question": "Which automakers grew strongly in Q1 2024 despite Tesla's weaker US performance?",
        "expected": [
            "Tesla sales and market share declined",
            "BMW, Cadillac, Ford, Hyundai, Kia, Lexus, Mercedes, Rivian, and VinFast had high growth",
            "Ford had the second-highest EV sales volume behind Tesla",
        ],
        "keywords": ["tesla", "bmw", "cadillac", "ford", "hyundai", "kia"],
    },
    {
        "question": "How are public and workplace chargers connected to high EV uptake in metro areas?",
        "expected": [
            "top uptake metros averaged much higher public chargers per million people",
            "top uptake metros also had more workplace chargers",
            "charging availability is linked to EV growth",
        ],
        "keywords": ["public chargers", "workplace chargers", "metropolitan", "uptake", "million", "charging"],
    },
    {
        "question": "What explains the mixed US consumer sentiment toward EVs?",
        "expected": [
            "Pew reports interest in EVs but also many people not likely to consider one",
            "charging confidence is limited",
            "environment and gas savings are major reasons for interested buyers",
        ],
        "keywords": ["pew", "38", "53", "environment", "gas", "confidence"],
    },
    {
        "question": "How could the Inflation Reduction Act and charging investments influence EV adoption by 2030?",
        "expected": [
            "IRA tax credits can reduce EV purchase costs",
            "charging infrastructure investments are expected to expand public chargers",
            "2030 adoption depends partly on charger availability",
        ],
        "keywords": ["inflation reduction act", "7500", "21 billion", "2030", "chargers", "tax credits"],
    },
    {
        "question": "Why are hybrids receiving more attention during the EV slowdown?",
        "expected": [
            "hybrid sales have accelerated while EV sales momentum slowed",
            "HEVs can have shorter payback periods",
            "multiple powertrains may support automaker earnings and investment",
        ],
        "keywords": ["hybrids", "hev", "phev", "payback", "powertrains", "slowdown"],
    },
    {
        "question": "How do Chinese EV makers affect the competitive outlook for US and global EV markets?",
        "expected": [
            "China has EV supply-chain and cost advantages",
            "Chinese makers are expanding abroad",
            "tariffs and policy responses affect US and European competition",
        ],
        "keywords": ["china", "exports", "tariffs", "cost", "supply chain", "europe"],
    },
    {
        "question": "What business opportunities are created by growth in the US EV charging market?",
        "expected": [
            "PwC forecasts large growth in charge points and EVs by 2030",
            "hardware, software, installers, and charge point operators are value pools",
            "CPOs can capture a larger share over time",
        ],
        "keywords": ["pwc", "27 million", "charge points", "hardware", "software", "cpo"],
    },
    {
        "question": "Which regions or states stand out in US charging infrastructure availability?",
        "expected": [
            "California has the most charger points",
            "New York and Texas are also high-ranking states",
            "charging is concentrated in metropolitan statistical areas",
        ],
        "keywords": ["california", "new york", "texas", "charger points", "metropolitan", "charging"],
    },
    {
        "question": "How do environment and fuel savings motives interact with infrastructure concerns?",
        "expected": [
            "interested consumers cite environment and gas savings",
            "public charging availability remains a major obstacle",
            "confidence in infrastructure predicts willingness to consider EVs",
        ],
        "keywords": ["environment", "gas", "public charging", "confidence", "consider", "infrastructure"],
    },
    {
        "question": "What does Ford and GM scaling back production targets imply about EV sentiment?",
        "expected": [
            "Ford and GM scaled back near-term production because demand was softer than forecasts",
            "they still plan to sell more EVs",
            "they remain committed to an electric future",
        ],
        "keywords": ["ford", "general motors", "scaling back", "demand", "committed", "electric future"],
    },
    {
        "question": "How do EV prices and incentives affect affordability in the US market?",
        "expected": [
            "Tesla price cuts lowered average transaction prices",
            "incentive spending increased",
            "tax credits and leasing can improve affordability",
        ],
        "keywords": ["tesla", "average transaction price", "incentives", "leasing", "7500", "affordability"],
    },
    {
        "question": "How do ChargePoint, Tesla Supercharger, and Electrify America differ in US charging networks?",
        "expected": [
            "ChargePoint leads total public charging ports",
            "Tesla dominates DC fast charging through Supercharger",
            "Electrify America is second in fast charging after Tesla",
        ],
        "keywords": ["chargepoint", "tesla", "supercharger", "electrify america", "dc fast", "ports"],
    },
    {
        "question": "How do pollution standards and policy signals affect EV investment decisions?",
        "expected": [
            "standards can give automakers and charging providers confidence to invest",
            "weakening standards can affect the EV demand narrative",
            "EVs support clean air, public health, and climate goals",
        ],
        "keywords": ["standards", "automakers", "confidence", "clean air", "public health", "climate"],
    },
    {
        "question": "How does the US EV market compare with global EV market trends?",
        "expected": [
            "EV markets differ by region",
            "China dominates global EV sales but other markets are growing",
            "some global markets are slowing while the US still shows mixed growth",
        ],
        "keywords": ["global", "china", "united states", "slowdown", "growth", "bnef"],
    },
]


@dataclass(frozen=True)
class Document:
    doc_id: str
    path: Path
    query: str
    title: str
    link: str
    snippet: str
    content: str


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    doc_id: str
    title: str
    link: str
    text: str
    path: Path


@dataclass(frozen=True)
class RagHit:
    score: float
    chunk: TextChunk
    reason: str


@dataclass(frozen=True)
class GraphAnswer:
    seed_entities: list[str]
    related_entities: list[str]
    relation_lines: list[str]
    hits: list[RagHit]
    context: str
    answer: str


@dataclass(frozen=True)
class BuildMetrics:
    load_seconds: float
    chunk_seconds: float
    flat_rag_seconds: float
    graph_build_seconds: float
    total_seconds: float
    corpus_chars: int
    chunk_chars: int


def normalize_for_match(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


NORMALIZED_ALIASES = {
    entity: sorted({normalize_for_match(alias) for alias in aliases + [entity]}, key=len, reverse=True)
    for entity, aliases in ENTITY_ALIASES.items()
}


def read_field(text: str, field: str) -> str:
    prefix = f"{field}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def load_documents(dataset_dir: Path = DATASET_DIR) -> list[Document]:
    docs = []
    paths = sorted(dataset_dir.glob("doc_*.txt"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    for path in paths:
        raw = path.read_text(encoding="utf-8", errors="replace")
        content = raw.split("Full Content:", 1)[1].strip() if "Full Content:" in raw else raw
        docs.append(
            Document(
                doc_id=path.stem,
                path=path,
                query=read_field(raw, "Query"),
                title=read_field(raw, "Title"),
                link=read_field(raw, "Link"),
                snippet=read_field(raw, "Snippet"),
                content=content,
            )
        )
    return docs


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 40]


def make_chunks(docs: Iterable[Document], max_chars: int = 1200, overlap: int = 1) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for doc in docs:
        sentences = split_sentences(doc.content)
        current: list[str] = []
        current_len = 0
        chunk_index = 0

        for sentence in sentences:
            if current and current_len + len(sentence) > max_chars:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{doc.doc_id}_chunk_{chunk_index}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        link=doc.link,
                        text=" ".join(current),
                        path=doc.path,
                    )
                )
                chunk_index += 1
                current = current[-overlap:] if overlap else []
                current_len = sum(len(item) for item in current)
            current.append(sentence)
            current_len += len(sentence)

        if current:
            chunks.append(
                TextChunk(
                    chunk_id=f"{doc.doc_id}_chunk_{chunk_index}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    link=doc.link,
                    text=" ".join(current),
                    path=doc.path,
                )
            )
    return chunks


def extract_entities(text: str) -> dict[str, int]:
    normalized = f" {normalize_for_match(text)} "
    entities: dict[str, int] = {}
    for entity, aliases in NORMALIZED_ALIASES.items():
        count = 0
        for alias in aliases:
            if alias:
                count += len(re.findall(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized))
        if count:
            entities[entity] = count
    return entities


def short_text(text: str, width: int = 220) -> str:
    return textwrap.shorten(re.sub(r"\s+", " ", text).strip(), width=width, placeholder=" ...")


class FlatRAG:
    def __init__(self, chunks: list[TextChunk], use_chroma: bool = True) -> None:
        self.chunks = chunks
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000)
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in chunks])
        self.collection = None
        if use_chroma:
            self.collection = self._build_chroma_collection()

    def _build_chroma_collection(self):
        try:
            import chromadb
            from chromadb.config import Settings
        except Exception:
            return None

        try:
            client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
            collection = client.create_collection(
                name=f"flat_rag_{uuid.uuid4().hex[:12]}",
                metadata={"hnsw:space": "cosine"},
            )
            embeddings = self.matrix.toarray().astype("float32")
            collection.add(
                ids=[chunk.chunk_id for chunk in self.chunks],
                documents=[chunk.text for chunk in self.chunks],
                metadatas=[
                    {
                        "doc_id": chunk.doc_id,
                        "title": chunk.title,
                        "link": chunk.link,
                        "path": str(chunk.path),
                    }
                    for chunk in self.chunks
                ],
                embeddings=embeddings.tolist(),
            )
            return collection
        except Exception:
            return None

    def query_vector(self, question: str):
        return self.vectorizer.transform([question])

    def retrieve(self, question: str, top_k: int = 5, allowed_doc_ids: set[str] | None = None) -> list[RagHit]:
        query_vec = self.query_vector(question)
        if allowed_doc_ids is None and self.collection is not None and query_vec.nnz:
            return self._retrieve_chroma(query_vec, top_k)

        candidate_indices = [
            index
            for index, chunk in enumerate(self.chunks)
            if allowed_doc_ids is None or chunk.doc_id in allowed_doc_ids
        ]
        if not candidate_indices:
            return []
        sims = cosine_similarity(query_vec, self.matrix[candidate_indices]).ravel()
        ranked = np.argsort(sims)[::-1][:top_k]
        return [
            RagHit(
                score=float(sims[rank]),
                chunk=self.chunks[candidate_indices[rank]],
                reason="tf-idf cosine similarity",
            )
            for rank in ranked
        ]

    def _retrieve_chroma(self, query_vec, top_k: int) -> list[RagHit]:
        result = self.collection.query(
            query_embeddings=query_vec.toarray().astype("float32").tolist(),
            n_results=top_k,
            include=["distances"],
        )
        hits: list[RagHit] = []
        for chunk_id, distance in zip(result["ids"][0], result["distances"][0]):
            chunk = self.chunk_by_id[chunk_id]
            score = 1.0 - float(distance) if distance is not None else 0.0
            hits.append(RagHit(score=score, chunk=chunk, reason="ChromaDB over TF-IDF embeddings"))
        return hits


class KnowledgeGraphRAG:
    def __init__(self, docs: list[Document], chunks: list[TextChunk], flat_rag: FlatRAG) -> None:
        self.docs = docs
        self.doc_by_id = {doc.doc_id: doc for doc in docs}
        self.chunks = chunks
        self.flat_rag = flat_rag
        self.graph = nx.Graph()
        self.doc_entities: dict[str, dict[str, int]] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        for doc in self.docs:
            doc_node = self._doc_node(doc.doc_id)
            searchable_text = f"{doc.query}\n{doc.title}\n{doc.snippet}\n{doc.content}"
            entities = extract_entities(searchable_text)
            self.doc_entities[doc.doc_id] = entities
            self.graph.add_node(
                doc_node,
                kind="document",
                doc_id=doc.doc_id,
                title=doc.title,
                link=doc.link,
                path=str(doc.path),
            )

            for entity, count in entities.items():
                self.graph.add_node(entity, kind="entity", label=entity)
                self.graph.add_edge(doc_node, entity, relation="mentions", weight=count)

            for left, right in itertools.combinations(sorted(entities), 2):
                if self.graph.has_edge(left, right):
                    self.graph[left][right]["weight"] += 1
                    self.graph[left][right]["docs"].add(doc.doc_id)
                else:
                    self.graph.add_edge(
                        left,
                        right,
                        relation="co_occurs_in_doc",
                        weight=1,
                        docs={doc.doc_id},
                    )

    @staticmethod
    def _doc_node(doc_id: str) -> str:
        return f"doc::{doc_id}"

    def query(self, question: str, top_k: int = 5) -> GraphAnswer:
        seed_entities = [entity for entity in extract_entities(question) if entity in self.graph]
        if not seed_entities:
            seed_entities = self._fallback_seed_entities(question)

        subgraph_nodes: set[str] = set(seed_entities)
        distances_by_seed: dict[str, dict[str, int]] = {}
        for seed in seed_entities:
            distances = nx.single_source_shortest_path_length(self.graph, seed, cutoff=2)
            distances_by_seed[seed] = distances
            subgraph_nodes.update(distances)

        document_nodes = [node for node in subgraph_nodes if self.graph.nodes[node].get("kind") == "document"]
        candidate_doc_ids = {self.graph.nodes[node]["doc_id"] for node in document_nodes}
        candidate_doc_ids.update(hit.chunk.doc_id for hit in self.flat_rag.retrieve(question, top_k=top_k))
        if not candidate_doc_ids:
            candidate_doc_ids = {hit.chunk.doc_id for hit in self.flat_rag.retrieve(question, top_k=top_k)}

        doc_scores = self._score_docs(seed_entities, candidate_doc_ids, distances_by_seed)
        hits = self.flat_rag.retrieve(question, top_k=max(top_k * 4, 12), allowed_doc_ids=candidate_doc_ids)
        reranked = self._rerank_graph_hits(hits, doc_scores, top_k)

        related_entities = self._related_entities(seed_entities, subgraph_nodes)
        relation_lines = self._relation_lines(seed_entities, related_entities, limit=10)
        context = self._textualize(question, seed_entities, related_entities, relation_lines, reranked)
        answer = answer_from_evidence(question, reranked, relation_lines, label="GraphRAG")
        return GraphAnswer(seed_entities, related_entities, relation_lines, reranked, context, answer)

    def _fallback_seed_entities(self, question: str) -> list[str]:
        hits = self.flat_rag.retrieve(question, top_k=3)
        counts: dict[str, int] = {}
        for hit in hits:
            for entity, count in self.doc_entities.get(hit.chunk.doc_id, {}).items():
                counts[entity] = counts.get(entity, 0) + count
        return [entity for entity, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:4]]

    def _score_docs(
        self,
        seed_entities: list[str],
        candidate_doc_ids: set[str],
        distances_by_seed: dict[str, dict[str, int]],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for doc_id in candidate_doc_ids:
            doc_node = self._doc_node(doc_id)
            score = 0.0
            entities = self.doc_entities.get(doc_id, {})
            matched_weight = 0.0
            total_weight = sum(ENTITY_WEIGHTS.get(seed, 1.0) for seed in seed_entities) or 1.0
            for seed in seed_entities:
                entity_weight = ENTITY_WEIGHTS.get(seed, 1.0)
                if seed in entities:
                    matched_weight += entity_weight
                    score += entity_weight * (1.0 + math.log1p(min(entities[seed], 5)))
                distance = distances_by_seed.get(seed, {}).get(doc_node)
                if distance:
                    score += entity_weight * (0.2 / distance)
            score += 2.0 * (matched_weight / total_weight)
            score += min(len(entities), 20) * 0.01
            scores[doc_id] = score
        return scores

    def _rerank_graph_hits(self, hits: list[RagHit], doc_scores: dict[str, float], top_k: int) -> list[RagHit]:
        reranked = []
        for hit in hits:
            graph_score = doc_scores.get(hit.chunk.doc_id, 0.0)
            combined = (hit.score * 10.0) + graph_score
            reranked.append(RagHit(score=combined, chunk=hit.chunk, reason=f"{hit.reason} + graph score"))
        sorted_hits = sorted(reranked, key=lambda hit: hit.score, reverse=True)

        diverse_hits: list[RagHit] = []
        seen_doc_ids = set()
        seen_titles = set()
        for hit in sorted_hits:
            if hit.chunk.doc_id in seen_doc_ids:
                continue
            title_key = normalize_for_match(hit.chunk.title)
            if title_key in seen_titles:
                continue
            seen_doc_ids.add(hit.chunk.doc_id)
            seen_titles.add(title_key)
            diverse_hits.append(hit)
            if len(diverse_hits) >= top_k:
                return diverse_hits

        for hit in sorted_hits:
            if hit in diverse_hits:
                continue
            diverse_hits.append(hit)
            if len(diverse_hits) >= top_k:
                break
        return diverse_hits

    def _related_entities(self, seed_entities: list[str], subgraph_nodes: set[str]) -> list[str]:
        candidates = [
            node
            for node in subgraph_nodes
            if self.graph.nodes[node].get("kind") == "entity" and node not in seed_entities
        ]
        return sorted(candidates, key=lambda node: self.graph.degree(node, weight="weight"), reverse=True)[:12]

    def _relation_lines(self, seed_entities: list[str], related_entities: list[str], limit: int) -> list[str]:
        lines = []
        for seed in seed_entities:
            for related in related_entities:
                if self.graph.has_edge(seed, related):
                    edge = self.graph[seed][related]
                    docs = sorted(edge.get("docs", []))
                    doc_label = ", ".join(docs[:3])
                    lines.append(f"{seed} -> {related} ({edge.get('relation')}; docs: {doc_label})")
        return lines[:limit]

    def _textualize(
        self,
        question: str,
        seed_entities: list[str],
        related_entities: list[str],
        relation_lines: list[str],
        hits: list[RagHit],
    ) -> str:
        lines = [
            f"Question: {question}",
            f"Seed entities: {', '.join(seed_entities) if seed_entities else 'none'}",
            f"2-hop related entities: {', '.join(related_entities) if related_entities else 'none'}",
            "Relations:",
        ]
        lines.extend([f"- {line}" for line in relation_lines] or ["- none"])
        lines.append("Evidence:")
        for hit in hits:
            lines.append(f"- [{hit.chunk.doc_id}] {hit.chunk.title}")
            lines.append(f"  Link: {hit.chunk.link}")
            lines.append(f"  Text: {short_text(hit.chunk.text, 550)}")
        return "\n".join(lines)


def answer_from_evidence(question: str, hits: list[RagHit], relation_lines: list[str] | None = None, label: str = "RAG") -> str:
    context = "\n\n".join(f"[{hit.chunk.doc_id}] {hit.chunk.title}\n{hit.chunk.text}" for hit in hits)
    llm_answer = maybe_llm_answer(question, context, label)
    if llm_answer:
        return llm_answer

    selected = select_relevant_sentences(question, hits, max_sentences=6)
    lines = [f"{label} extractive answer (LLM is disabled or unavailable):"]
    if relation_lines:
        lines.append("Graph relations used:")
        lines.extend(f"- {line}" for line in relation_lines[:4])
    lines.append("Evidence summary:")
    lines.extend(f"- {sentence}" for sentence in selected)
    return "\n".join(lines)


def get_config_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            registry_value, _ = winreg.QueryValueEx(key, name)
        return str(registry_value) if registry_value else None
    except OSError:
        return None


def clean_llm_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    if re.match(r"^\s*<think>", text, flags=re.IGNORECASE):
        parts = re.split(r"\n\s*\n", text, maxsplit=1)
        text = parts[1] if len(parts) > 1 else ""
    return text.strip()


def call_llm(user_prompt: str, system_prompt: str = "You are a careful RAG evaluator.") -> str | None:
    if LLM_DISABLED:
        return None
    api_key = get_config_value("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        base_url = get_config_value("OPENAI_BASE_URL")
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        model = get_config_value("OPENAI_MODEL") or "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        content = clean_llm_text(response.choices[0].message.content or "")
        return content or None
    except Exception:
        return None


def maybe_llm_answer(question: str, context: str, label: str) -> str | None:
    prompt = (
        "Answer the question in Vietnamese. Use only the supplied evidence. "
        "If evidence is insufficient, say what is missing. "
        "Do not include hidden reasoning or <think> tags; output only the final answer.\n\n"
        f"Mode: {label}\nQuestion: {question}\n\nEvidence:\n{context[:LLM_CONTEXT_CHARS]}"
    )
    return call_llm(prompt)


def maybe_llm_judge(question: str, expected: list[str], flat_answer_text: str, graph_answer_text: str) -> str:
    prompt = "\n".join(
        [
            "Ban la giam khao cho bai lab RAG.",
            "Dua tren expected evidence, hay cham cau tra loi Flat RAG va GraphRAG.",
            "Chi danh gia theo thong tin trong answer, khong tu bo sung kien thuc ngoai.",
            "Khong xuat <think>. Tra loi toi da 5 dong theo format:",
            "Flat RAG: x/5 - nhan xet ngan",
            "GraphRAG: x/5 - nhan xet ngan",
            "Better: Flat RAG / GraphRAG / Tie",
            "Hallucination risk: nhan xet ngan",
            "",
            f"Question: {question}",
            "Expected evidence:",
            *[f"- {item}" for item in expected],
            "",
            f"Flat RAG answer:\n{flat_answer_text[:LLM_JUDGE_ANSWER_CHARS]}",
            "",
            f"GraphRAG answer:\n{graph_answer_text[:LLM_JUDGE_ANSWER_CHARS]}",
        ]
    )
    return call_llm(prompt, system_prompt="You are a strict RAG judge. Final answer only.") or "LLM judge was not available."


def select_relevant_sentences(question: str, hits: list[RagHit], max_sentences: int = 6) -> list[str]:
    sentence_items: list[tuple[str, str]] = []
    for hit in hits:
        for sentence in split_sentences(hit.chunk.text):
            sentence_items.append((hit.chunk.doc_id, sentence))

    if not sentence_items:
        return []

    sentences = [item[1] for item in sentence_items]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000)
    matrix = vectorizer.fit_transform([question] + sentences)
    scores = cosine_similarity(matrix[0], matrix[1:]).ravel()
    ranked = np.argsort(scores)[::-1]

    selected: list[str] = []
    seen = set()
    for index in ranked:
        doc_id, sentence = sentence_items[index]
        normalized = normalize_for_match(sentence)
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(f"[{doc_id}] {sentence}")
        if len(selected) >= max_sentences:
            break
    return selected


def flat_answer(question: str, flat_rag: FlatRAG, top_k: int = 5) -> tuple[list[RagHit], str]:
    hits = flat_rag.retrieve(question, top_k=top_k)
    return hits, answer_from_evidence(question, hits, label="Flat RAG")


def source_list(hits: list[RagHit]) -> str:
    sources = []
    seen = set()
    for hit in hits:
        source_key = normalize_for_match(hit.chunk.title)
        if hit.chunk.doc_id in seen or source_key in seen:
            continue
        seen.add(hit.chunk.doc_id)
        seen.add(source_key)
        sources.append(f"{hit.chunk.doc_id}: {hit.chunk.title}")
    return "<br>".join(sources)


def keyword_coverage(text: str, keywords: list[str]) -> tuple[int, int]:
    normalized = normalize_for_match(text)
    found = sum(1 for keyword in keywords if normalize_for_match(keyword) in normalized)
    return found, len(keywords)


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def write_evaluation_report(
    flat_rag: FlatRAG,
    graph_rag: KnowledgeGraphRAG,
    output_path: Path,
    use_llm_judge: bool = False,
    csv_path: Path | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    csv_rows = []
    detail_sections = []
    for index, case in enumerate(EVALUATION_CASES, start=1):
        question = case["question"]
        flat_hits, flat_text = flat_answer(question, flat_rag)
        graph_result = graph_rag.query(question)

        flat_context = flat_text + "\n" + "\n".join(hit.chunk.text for hit in flat_hits)
        graph_context = graph_result.answer + "\n" + graph_result.context
        flat_cov = keyword_coverage(flat_context, case["keywords"])
        graph_cov = keyword_coverage(graph_context, case["keywords"])
        note = "GraphRAG has stronger entity coverage." if graph_cov[0] > flat_cov[0] else "Coverage is similar; inspect answer faithfulness."
        judge = maybe_llm_judge(question, case["expected"], flat_text, graph_result.answer) if use_llm_judge else ""
        flat_sources = source_list(flat_hits) or "none"
        graph_sources = source_list(graph_result.hits) or "none"

        if use_llm_judge:
            rows.append(
                "| {idx} | {question} | {flat_cov}/{flat_total} | {graph_cov}/{graph_total} | {note} | {judge} |".format(
                    idx=index,
                    question=question,
                    flat_cov=flat_cov[0],
                    flat_total=flat_cov[1],
                    graph_cov=graph_cov[0],
                    graph_total=graph_cov[1],
                    note=note,
                    judge=judge.replace("\n", "<br>"),
                )
            )
        else:
            rows.append(
                "| {idx} | {question} | {flat_cov}/{flat_total} | {graph_cov}/{graph_total} | {note} |".format(
                    idx=index,
                    question=question,
                    flat_cov=flat_cov[0],
                    flat_total=flat_cov[1],
                    graph_cov=graph_cov[0],
                    graph_total=graph_cov[1],
                    note=note,
                )
            )
        csv_rows.append(
            {
                "case": index,
                "question": question,
                "expected_evidence": " | ".join(case["expected"]),
                "flat_sources": flat_sources.replace("<br>", " | "),
                "flat_answer": flat_text,
                "flat_keyword_coverage": f"{flat_cov[0]}/{flat_cov[1]}",
                "graphrag_seed_entities": ", ".join(graph_result.seed_entities),
                "graphrag_sources": graph_sources.replace("<br>", " | "),
                "graphrag_answer": graph_result.answer,
                "graphrag_keyword_coverage": f"{graph_cov[0]}/{graph_cov[1]}",
                "llm_judge": judge,
                "note": note,
            }
        )

        detail_lines = [
            f"### Case {index}: {question}",
            "",
            "Expected evidence:",
            *[f"- {item}" for item in case["expected"]],
            "",
            "Flat RAG sources:",
            flat_sources,
            "",
            "Flat RAG answer:",
            "```text",
            flat_text,
            "```",
            "",
            "GraphRAG seed entities:",
            ", ".join(graph_result.seed_entities) or "none",
            "",
            "GraphRAG sources:",
            graph_sources,
            "",
            "GraphRAG answer:",
            "```text",
            graph_result.answer,
            "```",
        ]
        if use_llm_judge:
            detail_lines.extend(["", "LLM judge:", "```text", judge, "```"])
        detail_sections.append("\n".join(detail_lines))

    summary_header = "| # | Question | Flat keyword coverage | Graph keyword coverage | Note |"
    summary_separator = "|---|---|---:|---:|---|"
    if use_llm_judge:
        summary_header = "| # | Question | Flat keyword coverage | Graph keyword coverage | Note | LLM judge |"
        summary_separator = "|---|---|---:|---:|---|---|"
    llm_note = (
        "LLM answers and LLM judge are enabled when API calls succeed; otherwise the script falls back to extractive local answers."
        if use_llm_judge
        else "When LLM is disabled or unavailable, answers are extractive and therefore show retrieval quality rather than free-form LLM hallucination."
    )

    report = "\n".join(
        [
            "# Flat RAG vs GraphRAG Benchmark",
            "",
            f"This report compares a Flat RAG baseline with a 2-hop GraphRAG pipeline over {len(EVALUATION_CASES)} benchmark questions.",
            "Flat RAG uses ChromaDB when available over TF-IDF vectors; GraphRAG uses NetworkX over document/entity nodes.",
            "",
            "## Summary",
            "",
            summary_header,
            summary_separator,
            *rows,
            "",
            "Keyword coverage is a lightweight proxy, not a replacement for human grading.",
            llm_note,
            "",
            "## Details",
            "",
            *detail_sections,
            "",
        ]
    )
    output_path.write_text(report, encoding="utf-8")

    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)


def build_system(use_chroma: bool = True) -> tuple[list[Document], list[TextChunk], FlatRAG, KnowledgeGraphRAG]:
    docs, chunks, flat_rag, graph_rag, _ = build_system_with_metrics(use_chroma=use_chroma)
    return docs, chunks, flat_rag, graph_rag


def build_system_with_metrics(
    use_chroma: bool = True,
) -> tuple[list[Document], list[TextChunk], FlatRAG, KnowledgeGraphRAG, BuildMetrics]:
    started = time.perf_counter()

    load_started = time.perf_counter()
    docs = load_documents()
    load_seconds = time.perf_counter() - load_started

    chunk_started = time.perf_counter()
    chunks = make_chunks(docs)
    chunk_seconds = time.perf_counter() - chunk_started

    flat_started = time.perf_counter()
    flat_rag = FlatRAG(chunks, use_chroma=use_chroma)
    flat_rag_seconds = time.perf_counter() - flat_started

    graph_started = time.perf_counter()
    graph_rag = KnowledgeGraphRAG(docs, chunks, flat_rag)
    graph_build_seconds = time.perf_counter() - graph_started

    metrics = BuildMetrics(
        load_seconds=load_seconds,
        chunk_seconds=chunk_seconds,
        flat_rag_seconds=flat_rag_seconds,
        graph_build_seconds=graph_build_seconds,
        total_seconds=time.perf_counter() - started,
        corpus_chars=sum(len(doc.content) for doc in docs),
        chunk_chars=sum(len(chunk.text) for chunk in chunks),
    )
    return docs, chunks, flat_rag, graph_rag, metrics


def draw_knowledge_graph(graph_rag: KnowledgeGraphRAG, output_path: Path, max_entities: int = 32, max_docs: int = 28) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = graph_rag.graph
    entity_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "entity"]
    top_entities = sorted(entity_nodes, key=lambda node: graph.degree(node, weight="weight"), reverse=True)[:max_entities]

    doc_scores: dict[str, float] = {}
    for entity in top_entities:
        for neighbor in graph.neighbors(entity):
            if graph.nodes[neighbor].get("kind") == "document":
                doc_scores[neighbor] = doc_scores.get(neighbor, 0.0) + graph[entity][neighbor].get("weight", 1.0)
    top_docs = [node for node, _ in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:max_docs]]
    subgraph = graph.subgraph(top_entities + top_docs).copy()

    plt.figure(figsize=(18, 12))
    pos = nx.spring_layout(subgraph, seed=42, k=0.8)
    entity_subset = [node for node in subgraph if subgraph.nodes[node].get("kind") == "entity"]
    doc_subset = [node for node in subgraph if subgraph.nodes[node].get("kind") == "document"]

    nx.draw_networkx_edges(subgraph, pos, alpha=0.25, width=0.8, edge_color="#6b7280")
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=doc_subset,
        node_color="#93c5fd",
        node_size=260,
        alpha=0.85,
        label="Documents",
    )
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=entity_subset,
        node_color="#f97316",
        node_size=760,
        alpha=0.92,
        label="Entities",
    )
    labels = {
        node: graph.nodes[node].get("doc_id", node) if graph.nodes[node].get("kind") == "document" else node
        for node in subgraph.nodes
    }
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8)
    plt.title("Knowledge Graph: EV Dataset Entity-Document Network", fontsize=18)
    plt.legend(scatterpoints=1, loc="lower left")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_source_doc_ids(source_text: str) -> list[str]:
    return re.findall(r"\bdoc_\d+\b", source_text or "")


def parse_llm_winner(judge_text: str) -> str:
    normalized = normalize_for_match(judge_text or "")
    if "better graphrag" in normalized:
        return "GraphRAG"
    if "better flat rag" in normalized:
        return "Flat RAG"
    return "Tie"


def parse_score(judge_text: str, label: str) -> int | None:
    match = re.search(rf"{re.escape(label)}\s*:\s*(\d)\s*/\s*5", judge_text or "", flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def draw_llm_benchmark_graph(csv_path: Path, output_path: Path, max_doc_nodes: int = 28) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows: list[dict[str, str]]
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No benchmark rows found in {csv_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = nx.Graph()
    doc_counts: dict[str, int] = {}

    for row in rows:
        for doc_id in parse_source_doc_ids(row.get("flat_sources", "")):
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        for doc_id in parse_source_doc_ids(row.get("graphrag_sources", "")):
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    top_docs = {
        doc_id
        for doc_id, _ in sorted(doc_counts.items(), key=lambda item: item[1], reverse=True)[:max_doc_nodes]
    }

    graph.add_node("Flat RAG", kind="system")
    graph.add_node("GraphRAG", kind="system")
    graph.add_node("LLM Judge", kind="judge")

    for row in rows:
        case_id = row["case"]
        question_node = f"Q{case_id}"
        winner = parse_llm_winner(row.get("llm_judge", ""))
        flat_score = parse_score(row.get("llm_judge", ""), "Flat RAG")
        graph_score = parse_score(row.get("llm_judge", ""), "GraphRAG")

        graph.add_node(
            question_node,
            kind="question",
            winner=winner,
            label=f"Q{case_id}",
            question=row.get("question", ""),
        )
        graph.add_edge("LLM Judge", question_node, relation=winner)
        graph.add_edge("Flat RAG", question_node, relation="answered", score=flat_score or 0)
        graph.add_edge("GraphRAG", question_node, relation="answered", score=graph_score or 0)

        for doc_id in parse_source_doc_ids(row.get("flat_sources", "")):
            if doc_id not in top_docs:
                continue
            graph.add_node(doc_id, kind="document")
            graph.add_edge(question_node, doc_id, relation="flat_source")

        for doc_id in parse_source_doc_ids(row.get("graphrag_sources", "")):
            if doc_id not in top_docs:
                continue
            graph.add_node(doc_id, kind="document")
            graph.add_edge(question_node, doc_id, relation="graph_source")

        for entity in [item.strip() for item in row.get("graphrag_seed_entities", "").split(",") if item.strip()][:3]:
            entity_node = f"entity::{entity}"
            graph.add_node(entity_node, kind="entity", label=entity)
            graph.add_edge(question_node, entity_node, relation="seed_entity")

    plt.figure(figsize=(20, 13))
    pos = nx.spring_layout(graph, seed=11, k=0.9, iterations=120)

    question_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "question"]
    doc_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "document"]
    entity_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "entity"]
    system_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "system"]
    judge_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "judge"]

    question_colors = []
    for node in question_nodes:
        winner = graph.nodes[node].get("winner")
        if winner == "GraphRAG":
            question_colors.append("#22c55e")
        elif winner == "Flat RAG":
            question_colors.append("#ef4444")
        else:
            question_colors.append("#f59e0b")

    edge_colors = []
    edge_widths = []
    for _, _, data in graph.edges(data=True):
        relation = data.get("relation")
        if relation == "flat_source":
            edge_colors.append("#ef4444")
            edge_widths.append(0.8)
        elif relation == "graph_source":
            edge_colors.append("#22c55e")
            edge_widths.append(0.8)
        elif relation == "seed_entity":
            edge_colors.append("#8b5cf6")
            edge_widths.append(0.9)
        elif relation in {"Flat RAG", "GraphRAG", "Tie"}:
            edge_colors.append("#111827")
            edge_widths.append(1.3)
        else:
            edge_colors.append("#94a3b8")
            edge_widths.append(0.5)

    nx.draw_networkx_edges(graph, pos, alpha=0.35, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_nodes(graph, pos, nodelist=doc_nodes, node_color="#93c5fd", node_size=260, alpha=0.85)
    nx.draw_networkx_nodes(graph, pos, nodelist=entity_nodes, node_color="#a78bfa", node_size=480, alpha=0.9)
    nx.draw_networkx_nodes(graph, pos, nodelist=question_nodes, node_color=question_colors, node_size=560, alpha=0.95)
    nx.draw_networkx_nodes(graph, pos, nodelist=system_nodes, node_color="#0f172a", node_size=1200, alpha=0.95)
    nx.draw_networkx_nodes(graph, pos, nodelist=judge_nodes, node_color="#facc15", node_size=1100, alpha=0.95)

    labels = {}
    for node, data in graph.nodes(data=True):
        if data.get("kind") == "entity":
            labels[node] = data.get("label", node)
        else:
            labels[node] = data.get("label", node)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_color="#111827")

    legend = [
        Line2D([0], [0], marker="o", color="w", label="Question: GraphRAG wins", markerfacecolor="#22c55e", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Question: Flat RAG wins", markerfacecolor="#ef4444", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Question: Tie", markerfacecolor="#f59e0b", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Retrieved document", markerfacecolor="#93c5fd", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="GraphRAG seed entity", markerfacecolor="#a78bfa", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="LLM judge", markerfacecolor="#facc15", markersize=10),
    ]
    plt.legend(handles=legend, loc="lower left", fontsize=10)
    plt.title("LLM Benchmark Graph: Questions, Retrieved Evidence, GraphRAG Seeds, and Judge Outcomes", fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_cost_analysis(
    docs: list[Document],
    chunks: list[TextChunk],
    graph_rag: KnowledgeGraphRAG,
    metrics: BuildMetrics,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entity_count = sum(1 for _, data in graph_rag.graph.nodes(data=True) if data.get("kind") == "entity")
    doc_count = sum(1 for _, data in graph_rag.graph.nodes(data=True) if data.get("kind") == "document")
    corpus_tokens = estimate_tokens(" ".join(doc.content for doc in docs))
    chunk_tokens = estimate_tokens(" ".join(chunk.text for chunk in chunks))
    llm_context_tokens_per_answer = estimate_tokens("x" * LLM_CONTEXT_CHARS)
    llm_judge_tokens_per_case = estimate_tokens("x" * (LLM_JUDGE_ANSWER_CHARS * 2 + 1200))
    benchmark_count = len(EVALUATION_CASES)
    estimated_answer_tokens = benchmark_count * 2 * llm_context_tokens_per_answer
    estimated_judge_tokens = benchmark_count * llm_judge_tokens_per_case

    report = "\n".join(
        [
            "# Cost And Time Analysis",
            "",
            "## Build Time",
            "",
            "| Step | Seconds |",
            "|---|---:|",
            f"| Load {len(docs)} documents | {metrics.load_seconds:.3f} |",
            f"| Split into {len(chunks)} chunks | {metrics.chunk_seconds:.3f} |",
            f"| Build Flat RAG vector index | {metrics.flat_rag_seconds:.3f} |",
            f"| Build NetworkX knowledge graph | {metrics.graph_build_seconds:.3f} |",
            f"| Total startup/build time | {metrics.total_seconds:.3f} |",
            "",
            "## Graph Size",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Document nodes | {doc_count} |",
            f"| Entity nodes | {entity_count} |",
            f"| Edges | {graph_rag.graph.number_of_edges()} |",
            f"| Corpus characters | {metrics.corpus_chars} |",
            f"| Corpus tokens estimated | {corpus_tokens} |",
            f"| Chunk characters including overlap | {metrics.chunk_chars} |",
            f"| Chunk tokens estimated including overlap | {chunk_tokens} |",
            "",
            "## Token Usage Notes",
            "",
            "- Graph construction in this implementation uses local alias-based entity extraction, so API token usage for building the graph is 0.",
            "- Token counts are approximate, using characters / 4.",
            f"- LLM answer context is capped at about {llm_context_tokens_per_answer} prompt tokens per Flat/Graph answer.",
            f"- For {benchmark_count} benchmark questions, answer generation would use about {estimated_answer_tokens} prompt tokens before model output.",
            f"- If `--llm-judge` is enabled, judge prompts add about {estimated_judge_tokens} more prompt tokens.",
            "- Wall-clock time depends strongly on the configured model endpoint. The local graph build is fast; LLM calls dominate runtime.",
            "",
            f"Configured model: `{get_config_value('OPENAI_MODEL') or 'not set'}`",
            f"Configured base URL: `{get_config_value('OPENAI_BASE_URL') or 'not set'}`",
            "",
        ]
    )
    output_path.write_text(report, encoding="utf-8")


def print_stats(docs: list[Document], chunks: list[TextChunk], graph_rag: KnowledgeGraphRAG) -> None:
    entity_count = sum(1 for _, data in graph_rag.graph.nodes(data=True) if data.get("kind") == "entity")
    doc_count = sum(1 for _, data in graph_rag.graph.nodes(data=True) if data.get("kind") == "document")
    print(f"Documents: {len(docs)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Graph document nodes: {doc_count}")
    print(f"Graph entity nodes: {entity_count}")
    print(f"Graph edges: {graph_rag.graph.number_of_edges()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Flat RAG vs GraphRAG lab over the local EV dataset.")
    parser.add_argument("question", nargs="*", help="Question to ask. Omit when using --evaluate or --stats.")
    parser.add_argument("--mode", choices=["flat", "graph"], default="graph")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--evaluate", action="store_true", help="Run the 20-question benchmark.")
    parser.add_argument("--llm-judge", action="store_true", help="Use the configured LLM API to judge Flat RAG vs GraphRAG.")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM calls and use extractive local answers.")
    parser.add_argument("--deliverables", action="store_true", help="Generate all required lab outputs.")
    parser.add_argument("--graph-image", action="store_true", help="Generate the knowledge graph screenshot PNG.")
    parser.add_argument("--cost-analysis", action="store_true", help="Generate cost and build-time analysis.")
    parser.add_argument("--stats", action="store_true", help="Print corpus and graph statistics.")
    parser.add_argument("--no-chroma", action="store_true", help="Use pure sklearn retrieval instead of ChromaDB.")
    parser.add_argument("--output", default=str(REPORT_DIR / "benchmark_20.md"), help="Benchmark report path.")
    parser.add_argument("--csv-output", default=None, help="Benchmark CSV output path.")
    args = parser.parse_args()

    global LLM_DISABLED
    LLM_DISABLED = args.disable_llm

    docs, chunks, flat_rag, graph_rag, metrics = build_system_with_metrics(use_chroma=not args.no_chroma)

    if args.stats:
        print_stats(docs, chunks, graph_rag)

    if args.deliverables:
        benchmark_path = REPORT_DIR / "benchmark_20.md"
        csv_path = REPORT_DIR / "benchmark_20.csv"
        graph_path = REPORT_DIR / "knowledge_graph.png"
        cost_path = REPORT_DIR / "cost_analysis.md"
        write_evaluation_report(flat_rag, graph_rag, benchmark_path, use_llm_judge=args.llm_judge, csv_path=csv_path)
        draw_llm_benchmark_graph(csv_path, graph_path)
        write_cost_analysis(docs, chunks, graph_rag, metrics, cost_path)
        print(f"Wrote benchmark report: {benchmark_path}")
        print(f"Wrote benchmark CSV: {csv_path}")
        print(f"Wrote graph image: {graph_path}")
        print(f"Wrote cost analysis: {cost_path}")

    if args.evaluate:
        output_path = Path(args.output)
        csv_path = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
        write_evaluation_report(flat_rag, graph_rag, output_path, use_llm_judge=args.llm_judge, csv_path=csv_path)
        print(f"Wrote evaluation report: {output_path}")
        print(f"Wrote evaluation CSV: {csv_path}")

    if args.graph_image:
        graph_path = REPORT_DIR / "knowledge_graph.png"
        csv_path = REPORT_DIR / "benchmark_20.csv"
        if csv_path.exists():
            draw_llm_benchmark_graph(csv_path, graph_path)
        else:
            draw_knowledge_graph(graph_rag, graph_path)
        print(f"Wrote graph image: {graph_path}")

    if args.cost_analysis:
        cost_path = REPORT_DIR / "cost_analysis.md"
        write_cost_analysis(docs, chunks, graph_rag, metrics, cost_path)
        print(f"Wrote cost analysis: {cost_path}")

    if args.question:
        question = " ".join(args.question)
        if args.mode == "flat":
            hits, answer = flat_answer(question, flat_rag, top_k=args.top_k)
            print(answer)
            print("\nSources:")
            for hit in hits:
                print(f"- {hit.chunk.doc_id} ({hit.score:.3f}): {hit.chunk.title}")
        else:
            result = graph_rag.query(question, top_k=args.top_k)
            print(result.answer)
            print("\nSeed entities:")
            print(", ".join(result.seed_entities) or "none")
            print("\nSources:")
            for hit in result.hits:
                print(f"- {hit.chunk.doc_id} ({hit.score:.3f}): {hit.chunk.title}")

    if not any([args.evaluate, args.deliverables, args.graph_image, args.cost_analysis, args.stats, args.question]):
        parser.print_help()


if __name__ == "__main__":
    main()
