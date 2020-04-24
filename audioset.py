import datasetSQL
import csv
import h5py
import librosa
import soundfile
import sys, os

db1_path = "database/db_Audioset_16000.sqlite"
feature1_path = "database/mel_Audioset_16000.hdf5"

file1_csv = "meta/wavfiles.csv"
file2_csv = "meta/wavfiles2.csv"
class_csv = "meta/class_labels_indices.csv"
#wav_path = '/proj/asignal/Audioset/wav'
wav_path = '/home/zhaos/Data/Audioset/wav'
segment1_csv =  "meta/balanced_train_segments.csv"
segment2_csv =  "meta/eval_segments.csv"
ontology_json = "meta/ontology.json"

def initiate_database(db_path, segment_csv):
    db = datasetSQL.LabelSet(db_path)
    db.initialize()
    with open(segment_csv, 'r') as f:
        for item in csv.DictReader(f):
            segment = {}
            segment['segment_id'] = item['YTID']
            print (segment)
            db.__insert__(segment, 'segments')
    db.__commit__()

def link_audio_file(db_path, filecsv):
    db = datasetSQL.LabelSet(db_path)
    with open(filecsv, 'r') as f:
        #for item in csv.DictReader(f):
        for line in f.readlines():
            item = {}
            line = line.rstrip()
            item["YTID"] = line.split(',')[-1]
            item["filename"] = ','.join(line.split(',')[:-1]) + '.wav'

            sql = """
            UPDATE segments SET audio_file = '{0}'
            WHERE segment_id = '{1}'
            """.format(item['filename'].replace("'", "''"), item['YTID'])
            db.cursor.execute(sql)
        db.__commit__()

def add_classes(db_path, class_csv):
    db = datasetSQL.LabelSet(db_path)
    with open(class_csv, 'r') as f:
        for item in csv.DictReader(f):
            class_item = {}
            class_item['ASID'] = item['mid']
            class_item['class_name'] = item['display_name'].replace("'", "''")
            print (item)
            db.__insert__(class_item, 'classes')
        db.__commit__()

def add_labels(db_path, segment_csv):
    db = datasetSQL.LabelSet(db_path)
    with open(segment_csv, 'r') as f:
        for line in f.readlines():
            segment_id = line.split(',')[0]
            if segment_id == "YTID": #title line
                continue
            labels = ','.join(line.split(',')[3:]).replace('"', '').strip()
            for label in labels.split(','):
                sql = """
                INSERT INTO labels (segment_id, class_id, label_type)                
                SELECT segments.segment_id, classes.class_id, 0 FROM segments CROSS JOIN classes WHERE segments.segment_id = '{0}' AND classes.ASID = '{1}'
                """.format(segment_id, label)
                print (sql)
                db.cursor.execute(sql)
    db.__commit__()

def refine_labels(db_path):
    #Strategy: loop five times to ensure all the existing labels have their parental classes labeled.
    import json
    with open(ontology_json, 'r') as f:
        ontology_list = json.load(f)
    db = datasetSQL.LabelSet(db_path)

    sql = """
       ALTER TABLE classes
       ADD COLUMN leaf_node BOOL
    """
    try:
        db.cursor.execute(sql)
    except:
        pass
    
    for loop_i in range(5):
        for class_dict in ontology_list:
            parent_asid = class_dict['id']
            sql = """
            SELECT class_id from classes WHERE ASID = '{0}'
            """.format(parent_asid)
            db.cursor.execute(sql)
            record = db.cursor.fetchone()
            if record: #Some class in ontology does not exist in dataset
                parent_id = record[0]
            else:
                continue 

            if len(class_dict['child_ids']) == 0:
                sql = """
                UPDATE classes SET leaf_node=1
                WHERE class_id = {0}
                """.format(parent_id)
            else:
                sql = """
                UPDATE classes SET leaf_node=0
                WHERE class_id = {0}
                """.format(parent_id)

            db.cursor.execute(sql)
            
            for child_asid in class_dict['child_ids']:
                sql = """
                SELECT class_id from classes WHERE ASID = '{0}'
                """.format(child_asid)
                db.cursor.execute(sql)
                record = db.cursor.fetchone()
                if record: #Some class in ontology does not exist in dataset
                    child_id = record[0]
                else:
                    continue 
                print (parent_asid, child_asid)
                sql = """
                INSERT INTO labels (segment_id, class_id, label_type)
                SELECT segment_id, {1},0 from segments
                WHERE segment_id IN
                (SELECT segment_id FROM labels WHERE class_id = {0})
                AND segment_id NOT IN
                (SELECT segment_id FROM labels WHERE class_id = {1})                
                """.format(child_id, parent_id)
                db.cursor.execute(sql)
                #print (db.cursor.fetchall())

    
    db.__commit__()
    db.__close__()


    
def compute_features(db_path, feature_path, wav_root):
    h5w = h5py.File(feature_path, 'w')
    db = datasetSQL.LabelSet(db_path)
    #trg_sr = 48000
    trg_sr = 16000
    
    sql = """
    SELECT segment_id, audio_file FROM segments
    WHERE audio_file NOT NULL
    """
    segment_list = db.cursor.execute(sql)
    for segment_tuple in segment_list:
        segment_id, audio_file = segment_tuple[0].decode('utf-8'), segment_tuple[1].decode('utf-8')
        print (segment_id, audio_file)
        """
        y, src_sr = soundfile.read(os.path.join(wav_root, audio_file))
        if len(y.shape) > 1:
            y = y[:,0]
        y = librosa.core.resample(y, src_sr, trg_sr)
        """
        y = librosa.load(os.path.join(wav_root, audio_file), trg_sr)[0]
        try:
            #mel = librosa.feature.melspectrogram(y,trg_sr,n_fft=1024,hop_length=512,n_mels=128)
            mel = librosa.feature.melspectrogram(y,trg_sr,n_fft=1024,hop_length=500,n_mels=64)
            log_mel = librosa.power_to_db(mel).T
            print (log_mel.shape)
            h5w.create_dataset(segment_id, data=log_mel)

        except:
            print ("Failure:",segment_id, audio_file)
            sql = """
            UPDATE segments SET audio_file=NULL WHERE segment_id = '{0}'
            """.format(segment_id)
            db.cursor.execute(sql)
            db.__commit__()
            continue
    return


if __name__ == '__main__':
    initiate_database(db1_path, segment1_csv)
    initiate_database(db1_path, segment2_csv)    
    link_audio_file(db1_path, file1_csv)
    link_audio_file(db1_path, file2_csv)
    add_classes(db1_path, class_csv)
    add_labels(db1_path, segment1_csv)
    add_labels(db1_path, segment2_csv)
    compute_features(db1_path, feature1_path, wav_path)
    #refine_labels('database/db_Audioset_UrbanSound.sqlite')
