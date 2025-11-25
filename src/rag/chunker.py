import re
from typing import List, Dict, Optional
# from dataclasses import dataclass

# @dataclass
# class Chunk:
#     text: str
#     metadata: Dict
#     start_idx: int
#     end_idx: int
#     chunk_type: str # e.g., 'code', 'comment', 'docstring', 'text'

class DocumentChunker:
    def __init__(self,
                  chunk_size: int = 500, 
                  overlap: int = 50, 
                  min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.chars_per_token = 4

    def chunk_text(self, text:str, metadata: Optional[Dict] = None) -> List[Dict]:
        if not text or (len(text.strip()) < self.min_chunk_size):
            return []
        
        metadata = metadata or {}

        if self._is_markdown_heavy(text):
            return self._chunk_markdown(text, metadata)
        else:
            return self._chunk_plain_text(text, metadata)


    def _is_markdown_heavy(self, text: str) -> bool:
        
        markdown_patterns = [
            r'^#{1,6}\s',  # Headers
            r'```',         # Code blocks
            r'\*\*.*\*\*',  # Bold
            r'\[.*\]\(.*\)', # Links
        ]
        
        matches = sum(len(re.findall(pattern, text, re.MULTILINE))
                      for pattern in markdown_patterns)
    
        return matches > (len(text)/1000 * 5) # more than 5 markdown elements per 1000 chars
    
    def _chunk_markdown(self, text: str, metadata: Dict) -> List[Dict]:
        
        chunks = []
        pending_chunk = ""
        
        code_block_pattern = r'(```[\s\S]*?```)'
        parts = re.split(code_block_pattern, text)

        # current_section = ""
        current_header = ""

        # as we go through the parts, we need to keep track of headers
        for part in enumerate(parts):
            if not part[1].strip():
                continue

            # if this is a code block,
            if part[1].startswith('```'):
                # figure out the language
                lang_match = re.match(r'```(\w+)?', part[1])
                lang = lang_match.group(1) if lang_match else "code"
                # create a chunk and append to chunks
                chunk = {
                    "text": part[1],
                    "metadata": {
                        **metadata, 
                        "language": lang, 
                        "chunk_type": "code",
                        "header": current_header
                        }
                }
                chunks.append(chunk)

            # if it's not a code block,
            else:
                # split line by line and look for headers
                header_pattern = r'^(#{1,6})\s+(.+)$'
                lines = part[1].split('\n')

                # keep track of the content under each header
                current_chunk_lines = []

                # go line by line
                for line in lines:
                    # if we have a header,
                    header_match = re.match(header_pattern, line)
                    if header_match:
                        # save whatever we have so far
                        if current_chunk_lines:
                            chunk_text = '\n'.join(current_chunk_lines)
                            if len(chunk_text.strip()) > self.min_chunk_size:
                                chunks.append({
                                    'text': chunk_text,
                                    'metadata': {
                                        **metadata,
                                        "chunk_type": "text",
                                        "header": current_header
                                        }
                                })
                            # and reset
                            current_chunk_lines = []
                        # update current header
                        # level = len(header_match.group(1))
                        current_header = header_match.group(2).strip()
                        current_chunk_lines.append(line)
                    else:
                        current_chunk_lines.append(line)

                        current_text = '\n'.join(current_chunk_lines)

                        if len(current_text) >= self.chunk_size * self.chars_per_token:
                            split_point = self._find_split_point()
                            if split_point > 0:
                                chunk_text = current_text[:split_point]
                                chunks.append({
                                    'text': chunk_text,
                                    'metadata': {
                                        **metadata,
                                        "chunk_type": "text",
                                        "header": current_header
                                        }
                                })
                                # reset current_chunk_lines to the remaining text
                                remaining = current_text[split_point - self.overlap:]
                                current_chunk_lines = remaining.split('\n')

                if current_chunk_lines:
                    chunk_text = '\n'.join(current_chunk_lines)
                    
                    if len(chunk_text.strip()) < self.min_chunk_size:
                        if pending_chunk:
                            pending_chunk += "\n\n" + chunk_text
                        else:
                            pending_chunk = chunk_text
                            if len(pending_chunk) > self.min_chunk_size:
                                chunks.append({
                                    'text': chunk_text,
                                    'metadata': {
                                        **metadata,
                                        "chunk_type": "text",
                                        "header": current_header
                                        }
                                })
                                pending_chunk = ""

                    else:
                        if pending_chunk:
                            chunk_text = pending_chunk + "\n\n" + chunk_text
                            pending_chunk = ""

                        chunks.append({
                            'text': chunk_text,
                            'metadata': {
                                **metadata,
                                'chunk_type': 'text',
                                'header': current_header
                            }
                        })

        if pending_chunk:
            chunks.append({
                'text': pending_chunk,
                'metadata': {
                    **metadata,
                    "chunk_type": "text",
                    "header": current_header
                    }
            })

        return chunks 

    def _chunk_plain_text(self, text: str, metadata: Dict) -> List[Dict]:
        chunks = []

        paragraphs = text.split('\n\n')
        current_chunk = ""

        for para in paragraphs:
            if not para.strip():
                continue
            if len(current_chunk + para) > self.chunk_size * self.chars_per_token:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            **metadata,
                            "chunk_type": "text"
                            }
                    })
                current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    **metadata,
                    "chunk_type": "text"
                    }
            })

        return chunks
        
    
    def _find_split_point(self, text: str) -> int:
        # to implement: find best split point in text
        target = self.chunk_size * self.chars_per_token
        para_break = text.find('\n\n', int(target * 0.8))

        if para_break != -1 and para_break < target * 1.2:
            return para_break
        
        sentence_endings = ['. ', '! ', '? ', '.\n', '?\n', '!\n']
        best_pos = -1

        for ending in sentence_endings:
            pos = text.rfind(ending, int(target*0.5), int(target*1.2))
            if pos > best_pos:
                best_pos = pos + len(ending)

        if best_pos > 0:
            return best_pos
        
        space_pos = text.rfind(' ', int(target*0.8), target)
        if space_pos != -1:
            return space_pos
        
        return target

  
    def chunk_code(self, code: str, language: str = "javascript") -> List[Dict]:
        chunks = []
        
        if language in ["javascript", "typescript"]:
            patterns = [
                r'((?:export\s+)?(?:async\s+)?function\s+\w+[\s\S]*?\n})',  # Functions
                r'((?:export\s+)?class\s+\w+[\s\S]*?\n})',  # Classes  
                r'(const\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>\s*{[\s\S]*?\n})',  # Arrow functions
            ]
        
            remaining_code = code

            for pattern in patterns:
                matches = re.finditer(pattern, remaining_code, re.MULTILINE)
                
                for match in matches:
                    func_code = match.group(0)
                    name_match = re.search(r'(function|class|const)\s+(\w+)', func_code)
                    name = name_match.group(1) if name_match else "unknown"

                    chunks.append({
                        'text': func_code,
                        'metadata': {
                            "language": language,
                            "name": name,
                            "chunk_type": "code"
                        }
                    })

        return chunks if chunks else [{
              'text': code,
              'metadata': {
                  "language": language,
                  "chunk_type": "code"
              }}]