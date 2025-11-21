# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Text post-processing utilities for streaming ASR.
"""

def remove_char_duplicates(text: str, max_repeat: int = 2) -> str:
    """
    Remove excessive character repetitions from text.
    
    This is a workaround for LLM models that don't have repetition penalty,
    which can cause character-level duplicates like "特特定性" -> "特定性".
    
    Args:
        text: Input text with potential duplicates
        max_repeat: Maximum allowed consecutive repetitions (default: 2)
        
    Returns:
        Text with duplicates removed
        
    Examples:
        >>> remove_char_duplicates("特特定性")
        '特定性'
        >>> remove_char_duplicates("成成在在這這些")
        '成在這些'
        >>> remove_char_duplicates("我我我我")
        '我'
    """
    if not text:
        return text
    
    result = []
    i = 0
    
    while i < len(text):
        char = text[i]
        result.append(char)
        
        # Count consecutive repetitions
        count = 1
        j = i + 1
        while j < len(text) and text[j] == char:
            count += 1
            j += 1
        
        # Skip excessive repetitions
        if count > max_repeat:
            i = j  # Skip all repetitions
        else:
            # Keep up to max_repeat
            for _ in range(min(count - 1, max_repeat - 1)):
                result.append(char)
            i = j
    
    return ''.join(result)


def remove_word_duplicates(text: str) -> str:
    """
    Remove consecutive duplicate words from text.
    
    Args:
        text: Input text with potential duplicate words
        
    Returns:
        Text with duplicate words removed
        
    Examples:
        >>> remove_word_duplicates("上帝 上帝 的 道")
        '上帝 的 道'
    """
    if not text:
        return text
    
    words = text.split()
    if not words:
        return text
    
    result = [words[0]]
    for i in range(1, len(words)):
        if words[i] != words[i-1]:
            result.append(words[i])
    
    return ' '.join(result)


def clean_asr_output(text: str) -> str:
    """
    Clean ASR output by removing various types of duplicates.
    
    Args:
        text: Raw ASR output
        
    Returns:
        Cleaned text
    """
    # Remove newlines and extra whitespace
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # First remove character-level duplicates
    text = remove_char_duplicates(text, max_repeat=1)
    
    # Then remove word-level duplicates
    text = remove_word_duplicates(text)
    
    # Clean up multiple spaces
    import re
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
