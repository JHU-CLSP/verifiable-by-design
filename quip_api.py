import requests
from typing import Optional, List, Dict
from time import sleep
from tqdm import tqdm

def quip_api(texts: List[str], format_quotes=False) -> Optional[List[Dict]]:
    # url = 'http://localhost:8009/quip'
    # url = 'https://acc2-private-wiki.dataportraits.org/quip'
    # url = 'https://cnn-dailymail-quip.dataportraits.org/quip'
    url = 'http://localhost:8566/quip'
    data = {'documents': texts, 'format_quotes': format_quotes}
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, json=data, headers=headers)
    except Exception as e:
        print("QUIP API Request failed with exception: ", e)
        return None
    if response.status_code == 200:
        response_json = response.json()
        quip_stats = [item for item in response_json]
        if len(quip_stats) != len(texts):
            print(f"QUIP API Request failed: expected {len(texts)} responses, got {len(quip_stats)}")
            # breakpoint()
            return None
        return quip_stats
    else:
        print(f"QUIP API Request failed with status code: {response.status_code}")
        return None
    
def get_quoted_segments(quip_stat, return_idx=False) -> List[str] | List[list]:
    '''
    returns quoted segments from the preprocessed doc in a list

    if doc is too short, an empty list is returned

    return_idx: if True, return the start and end index of each quoted segments; index is defined as the index of the first character in the raw quoted segment (25gram)
    '''
    ret = []
    if quip_stat['quip_report']['too_short']:
        return ret
    for i in range(len(quip_stat['segments'])):
        if quip_stat['is_member'][i]:
            if i==0 or quip_stat['is_member'][i-1]==False:
                if return_idx:
                    ret.append([i, i+1])
                else:
                    ret.append(quip_stat['segments'][i])
            else: # quip_stat['is_member'][i-1]==True, there must already be a segment in ret
                if return_idx:
                    ret[-1][-1] = i
                else:
                    ret[-1] += quip_stat['segments'][i][-1]
    return ret

def batch_quip_api(texts: List[str], batch_size=100, max_trials=5, wait_between_trials=2, simple_report_format=False) -> Optional[List[Dict]]:
    '''
    quip api with batching and retries
    '''
    quip_stats = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Batch'):
        batch_texts = texts[i:i+batch_size]
        success = False
        trials = 0
        while not success and trials < max_trials:
            batch_quip_stats = quip_api(batch_texts)
            if batch_quip_stats:
                success = True
                break
            else:
                sleep(wait_between_trials)
            trials += 1
        if success:
            quip_stats.extend(batch_quip_stats)
            print(f'Processed {len(batch_texts)} texts, {len(quip_stats)} total')
        else:
            print('Max trials reached, using Nones as placeholders')
            quip_stats.extend(len(batch_texts)*[None])
    if simple_report_format:
        reports = []
        for quip_stat in quip_stats:
            # quip_stats may be None
            if isinstance(quip_stat, dict) and 'quip_report' in quip_stat:
                quip_report = quip_stat['quip_report']
                quip_report['quoted_segments'] = get_quoted_segments(quip_stat)
                quip_report['longest_segment_string_length'] = max([len(segment) for segment in quip_report['quoted_segments']]) if len(quip_report['quoted_segments']) > 0 else 0
                reports.append(quip_report)
            else:
                reports.append(None)
        return reports
    else:
        return quip_stats


def process_dataset_main():
    import argparse
    from datasets import load_dataset, DatasetDict

    parser = argparse.ArgumentParser(description='Run QUIP API on a huggingface dataset')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--response_key', type=str, default='response')
    parser.add_argument('--output_dir', type=str, default='data/dataset_quip')
    args = parser.parse_args()

    dataset_dict = load_dataset(args.dataset_name)
    new_ddict = {}
    for split, ds in dataset_dict.items():
        print(f"Processing {split} split")
        quip_stats = batch_quip_api(ds[args.response_key])
        # add column to dataset
        new_ds = ds.add_column('quip_stats', quip_stats)
        new_ddict[split] = new_ds
    new_ddict = DatasetDict(new_ddict)
    new_ddict.save_to_disk(args.output_dir + '/' + args.dataset_name)

if __name__ == '__main__':
    
    example = batch_quip_api(["This is a quote from Wikipedia: Jellyfish are mainly free-swimming marine animals with umbrella-shaped bells"], simple_report_format=True)
    print(example[0]['quip_report'])
