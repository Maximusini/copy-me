import polars as pl
import json
import yaml
import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

def sanitize_urls(text):
    """
    Находит все URL в тексте и удаляет из них параметры запроса и фрагменты.
    Превращает длинные ссылки в чистые короткие адреса.
    """
    if not text:
        return ""
    
    url_pattern = r'https?://[^\s]+'
    
    def clean_match(match):
        url = match.group(0)
        try:
            parsed = urlparse(url)
            cleaned = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
            return cleaned
        except Exception:
            return url
            
    return re.sub(url_pattern, clean_match, text)

def run_preprocessing():
    config_path = Path('config.yaml')
    if not config_path.exists():
        raise FileNotFoundError('[!] Create config.yaml based on config.example.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    your_id = str(config['user_id'])
    your_username = str(config['username'])
    window_size = int(config['window_size'])
    telegram_export_path = str(config['telegram_export_path'])
    output_dataset_path = str(config['output_dataset_path'])
    system_prompt = str(config['system_prompt'])
    session_gap_limit = int(config.get('session_gap_limit', 18000))
    
    clean_your_id = your_id.replace("user", "").replace("channel", "").replace("chat", "")
    
    print('[*] Loading and parsing Telegram JSON...')
    with open(telegram_export_path, 'r', encoding='utf-8') as f:
        tg_json = json.load(f)
        
    data = []
    for message in tg_json.get('messages', []):
        msg_type = message.get('type')
        date = message.get('date')
        
        sender = message.get('from_id') or message.get('from') or message.get('actor_id') or message.get('actor', 'Unknown')
        
        if isinstance(sender, dict):
            sender = str(
                sender.get('id') or 
                sender.get('user_id') or 
                sender.get('channel_id') or 
                sender.get('chat_id') or 
                'Unknown'
            )
        else:
            sender = str(sender)
        
        clean_sender = sender.replace("user", "").replace("channel", "").replace("chat", "")
        
        if clean_sender == clean_your_id or sender == your_username:
            role = 'gpt'
        else:
            role = 'human'
        
        raw_text = message.get('text', '')
        if isinstance(raw_text, str):
            text = raw_text
        elif isinstance(raw_text, list):
            text = ''.join(
                part.get('text', '') if isinstance(part, dict) else str(part)
                for part in raw_text
            )
        else:
            text = ''
            
        text = sanitize_urls(text)
        
        data.append({
            "msg_type": msg_type,
            "date": date,
            "role": role,
            "text": text
        })
        
    df = pl.DataFrame(data)
    df = df.with_columns(pl.col('date').str.to_datetime())
    df = df.filter((pl.col('text') != '') & (pl.col('msg_type') == 'message'))
    
    df = df.sort('date')    
    df = df.with_columns(pl.col('date').diff().alias('time_delta'))
    
    df = df.with_columns(
        pl.when(pl.col('time_delta').dt.total_seconds() > session_gap_limit)
        .then(True)
        .otherwise(False)
        .fill_null(True)
        .alias('session_start')
    )
    df = df.with_columns(pl.col('session_start').cast(pl.Int64).cum_sum().alias('session_id'))
    
    df = df.with_columns(
        pl.when(
            (pl.col('role') != pl.col('role').shift()) | 
            (pl.col('session_id') != pl.col('session_id').shift())
        )
        .then(True)
        .otherwise(False)
        .fill_null(True)
        .alias('turn_start')
    )
    df = df.with_columns(pl.col('turn_start').cast(pl.Int64).cum_sum().alias('turn_id'))
    
    grouped_df = df.group_by(pl.col('turn_id')).agg(
        pl.col('session_id').first(),
        pl.col('role').first(),
        pl.col('text').sort_by('date').str.join('\n\n'),
        pl.col('date').sort_by('date').first()
    ).sort('date')
    
    sessions_dict = {}
    for row in grouped_df.to_dicts():
        sid = row['session_id']
        if sid not in sessions_dict:
            sessions_dict[sid] = []
        sessions_dict[sid].append(row)
        
    dataset = []
    
    chunk_size = window_size if window_size % 2 == 0 else window_size - 1
    if chunk_size < 2:
        chunk_size = 2
    
    for sid, turns in sessions_dict.items():
        for start in range(0, len(turns), chunk_size):
            chunk = turns[start:start + chunk_size]
            
            while chunk and chunk[0]['role'] == 'gpt':
                chunk = chunk[1:]
                
            while chunk and chunk[-1]['role'] == 'human':
                chunk = chunk[:-1]
                
            if len(chunk) < 2:
                continue
                
            curr_conv = [{"from": "system", "value": system_prompt}]
            for msg in chunk:
                curr_conv.append({"from": msg['role'], "value": msg['text']})
                
            dataset.append({"conversations": curr_conv})
    
    output_path = Path(output_dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f'[*] The dataset has been successfully created. Total examples: {len(dataset)}')

if __name__ == '__main__':
    run_preprocessing()