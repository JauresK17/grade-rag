from typing import List
import logfire


def _hard_split(text: str, chunk_size: int) -> List[str]:
    """Découpe brutalement un texte trop long en tranches de chunk_size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """
    Chunker par paragraphes. Tout paragraphe plus grand que chunk_size
    est lui-même redécoupé pour ne jamais dépasser la limite.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            # Si le paragraphe seul dépasse la taille, on le redécoupe en dur
            if len(p) >= chunk_size:
                # on ferme d'abord le chunk courant
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # puis on ajoute les tranches du gros paragraphe
                chunks.extend(_hard_split(p, chunk_size))
                continue

            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks