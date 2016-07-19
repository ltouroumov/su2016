import csv, sys
from pprint import pprint

class Dataset(object):
    def __init__(self, filename, columns):
        self.filename = filename
        self.columns = columns
        self.data = None
    
    def load(self, collectors, skip_collect=False):
        with open(self.filename, "r") as fd:
            csv_reader = csv.DictReader(fd, fieldnames=self.columns, delimiter=";")
            _header = next(csv_reader)
            raw_data = []
            if skip_collect:
                raw_data = [row for row in csv_reader]
            else:
                for row in csv_reader:
                    for column in self.columns:
                        collectors[column].collect(row[column], row)
                    
                    raw_data.append(row)
            
            self.data = [{key: collectors[key].transform(value) if key in collectors else value for key, value in row.items()} for row in raw_data]
    
    def extract(self, columns):
        return [[row[column] for column in columns] for row in self.data]

class LambdaCollector(object):
    def __init__(self, transform):
        self._transform = transform

    def collect(self, value, row):
        pass
    
    def transform(self, value):
        return self._transform(value)

class PassCollector(object):
    def collect(self, value, row):
        pass
    
    def transform(self, value):
        return value

class NumberCollector(object):
    def __init__(self, ctype):
        self.ctype = ctype
        self.count = 0
        self.sum = 0

    def collect(self, value, row):
        pass
        # if value not in ("", "NR"):
        #     self.count += 1
        #     self.sum += self.ctype(value)
    
    def transform(self, value):
        if value in ("", "NR"):
            return self.ctype(-1)
        else:
            return self.ctype(value)
        # if value in ("", "NR"):
        #     return self.sum / self.count
        # else:
        #     return self.ctype(value)

class AverageDelegate(object):
    def __init__(self, cname, ctype):
        self.cname = cname
        self.ctype = ctype
        self.count = 0
        self.sum = 0
    
    def update(self, row):
        value = row[self.cname]

        if value not in ("", "NR"):
            self.count += 1
            self.sum += self.ctype(value)
    
    def value(self):
        return self.sum / self.count

class CountDelegate(object):
    def __init__(self):
        self.count = 0
    
    def update(self, row):
        self.count += 1
    
    def value(self):
        return self.count

class StringCollector(object):
    def __init__(self, delegate):
        self.delegate = delegate
        self.types = {}
    
    def update(self, key, row):
        if key not in self.types:
            self.types[key] = self.delegate()
        
        self.types[key].update(row)
    
    def average(self, key):
        if key not in self.types:
            return -1
        else:
            return self.types[key].value()

    def collect(self, value, row):
        self.update(value, row)
    
    def transform(self, value):
        return self.average(value)

def process_dataset(infile, outfile, collectors, skip_collect=False, drop_columns=[]):
    ds = Dataset(infile, columns = [column for column in [
        'id',
        'annee_naissance',
        'annee_permis',
        'marque',
        'puis_fiscale',
        'anc_veh',
        'codepostal',
        'energie_veh',
        'kmage_annuel',
        'crm',
        'profession',
        'var1',
        'var2',
        'var3',
        'var4',
        'var5',
        'var6',
        'var7',
        'var8',
        'var9',
        'var10',
        'var11',
        'var12',
        'var13',
        'var14',
        'var15',
        'var16',
        'var17',
        'var18',
        'var19',
        'var20',
        'var21',
        'var22',
        'prime_tot_ttc'
    ] if column not in drop_columns])

    ds.load(collectors, skip_collect)

    non_features = ['codepostal', 'marque', 'energie_veh', 'profession', 'annee_permis', 'var1', 'var2', 'var6', 'var8', 'var10', 'var14', 'var16']
    features = [feature for feature in ds.columns if feature not in non_features]

    with open(outfile, mode="w+") as fd:
        csv_writer = csv.writer(fd, delimiter=';')
        csv_writer.writerow(features)
        for row in ds.extract(columns=features):
            csv_writer.writerow(row)


if __name__ == "__main__":
    collectors={
        'id': LambdaCollector(transform=int),
        'annee_naissance': NumberCollector(int),
        'annee_permis': NumberCollector(int),
        'marque': PassCollector(),
        'puis_fiscale': NumberCollector(int),
        'anc_veh': NumberCollector(int),
        'codepostal': PassCollector(),
        'energie_veh': PassCollector(),
        'kmage_annuel': NumberCollector(int),
        'crm': NumberCollector(int),
        'profession': PassCollector(),
        'var1': NumberCollector(int),
        'var2': NumberCollector(int),
        'var3': NumberCollector(int),
        'var4': NumberCollector(int),
        'var5': NumberCollector(int),
        'var6': StringCollector(delegate=CountDelegate),
        'var7': NumberCollector(int),
        'var8': PassCollector(),
        'var9': NumberCollector(int),
        'var10': NumberCollector(int),
        'var11': NumberCollector(int),
        'var12': NumberCollector(int),
        'var13': NumberCollector(int),
        'var14': PassCollector(),
        'var15': NumberCollector(int),
        'var16': NumberCollector(int),
        'var17': NumberCollector(int),
        'var18': NumberCollector(int),
        'var19': NumberCollector(int),
        'var20': NumberCollector(int),
        'var21': NumberCollector(int),
        'var22': NumberCollector(int),
        'prime_tot_ttc': NumberCollector(float)
    }

    process_dataset(infile="./data/ech_apprentissage.csv",
                    outfile='./data/ech_apprentissage_clean.csv',
                    collectors=collectors)

    process_dataset(infile='./data/ech_test.csv',
                    outfile='./data/ech_test_clean.csv',
                    collectors=collectors,
                    skip_collect=True,
                    drop_columns=['prime_tot_ttc'])