import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.pipelines.data_pipeline import DataPipeline

def predict_rest_mex_test(model: nn.Module, pipeline: DataPipeline, output_filename: str,
                        device: str, batch_size: int = 64):
    model = model.to(device).eval()
    try:
        root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
        test_path = os.path.join(root_path, 'data', 'Test_DataSet', 'Rest-Mex_2025_test.xlsx')
    except FileNotFoundError:
        test_path = '/content/Rest-Mex_2025_test.xlsx'

    df = pd.read_excel(test_path)
    df['Review'] = df['Title'].astype(str) + ' ' + df['Review'].astype(str)
    df['Review'] = df['Review'].str.lower()
    fix_map = {
            'ãº': 'ú',
            'ã¡': 'á',
            'ã!': 'ó',
            'ã©': 'é',
            'ã³': 'ó',
            'ã²': 'ó',
            'ã±': 'ñ',
            'â¡': '¡',
            'å': 'a',
            'ê': 'e',
            'ã': 'í',
            '½': 'medio',
            '¼': 'cuarto',
            ':))': ' feliz ',
            ':)': ' feliz ',
            ':(': ' triste ',
            '>:(': ' mal ',
            ' :/': ' confuso ',
            ':-)': ' feliz ',
            '\x85':''
        }
    symbols_to_remove = r'[,!¡¿\?%\.\(\)/\\_\-¢£¦¤¨©ª«¬®°²¹º»×æ÷·&"\'~*\{\}\|\+\@\[\\\]\=\`<>\:\;]'
    def fix_encoding_issues(text):
            for bad in fix_map:
                text = text.replace(bad, fix_map[bad])
            return text

    df["Review"] = df["Review"].apply(fix_encoding_issues)
    reviews = df["Review"].str.replace(symbols_to_remove, '', regex=True)
        
    ids = df['ID'].tolist()

    # 2. Procesar reseñas completas
    input_ids_list, lengths_list = [], []
    for text in reviews:
        tokens = pipeline.processor.process(text)
        ids_ = [pipeline.vocab.get(t, pipeline.vocab['unk']) for t in tokens]
        length = len(ids_)
        input_ids_list.append(torch.tensor(ids_, dtype = torch.long))
        lengths_list.append(length)

    # 3. Dataset tipo DataLoader agrupado por instancia
    dataset = list(zip(input_ids_list, lengths_list))

    def collate_test_fn(batch):
        input_ids_batch, lengths_batch = zip(*batch)
        max_len = max(len(ids) for ids in input_ids_batch)

        padded_input_ids = [
            F.pad(ids, (0, max_len - len(ids)), value = pipeline.vocab['pad']) for ids in input_ids_batch
        ]
        padded_input_ids = torch.stack(padded_input_ids).unsqueeze(1) # (B, 1, T)
        lengths_tensor = torch.tensor(lengths_batch).unsqueeze(1)     # (B, 1)
        review_mask = torch.ones(len(batch), 1, dtype = torch.bool)   # (B, 1)

        return padded_input_ids.to(device), lengths_tensor.to(device), review_mask.to(device)

    loader = DataLoader(dataset, batch_size = batch_size, collate_fn = collate_test_fn)

    predictions = []

    with torch.no_grad():
        for input_ids, lengths, review_mask in tqdm(loader, desc = "Predicting Test Set", colour = "cyan"):
            polarity_logits, town_logits, type_logits = model(input_ids, lengths, review_mask)

            p_pred = polarity_logits.argmax(dim = -1).squeeze(1).cpu().tolist()
            town_pred = town_logits.argmax(dim = -1).cpu().tolist()
            t_pred = type_logits.argmax(dim = -1).squeeze(1).cpu().tolist()

            for p, m, t in zip(p_pred, town_pred, t_pred):
                polarity = p + 1  # Shift back to 1–5
                town = pipeline.id2town.get(m, "Unknown")
                place_type = pipeline.id2type.get(t, "Unknown")
                predictions.append((polarity, town, place_type))

    # 5. Guardar predicciones en formato oficial
    with open(output_filename, "w") as f:
        for idx, (pol, town, typ) in zip(ids, predictions):
            f.write(f"rest-mex\t{idx}\t{pol}\t{town}\t{typ}\n")

    print(f"Archivo generado: {output_filename}")