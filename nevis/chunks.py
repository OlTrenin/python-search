def chunks(content, tokenizer):
    result = []
    for paragraph in content.splitlines():
        offsets = tokenizer.encode(paragraph, add_special_tokens=False).offsets
        for start in range(0, len(offsets), 224):
            window = offsets[start:start + 256]
            result.append(paragraph[window[0][0]:window[-1][1]])
            if start + 256 >= len(offsets):
                break
    if len(result) > 512:
        raise ValueError('Document exceeds the limit of 512 semantic chunks')
    return result
