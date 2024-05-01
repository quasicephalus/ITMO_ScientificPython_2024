import subprocess
from Bio import SeqIO
import requests
import re


def get_uniprot(ids: list):
  accessions = ','.join(ids)
  endpoint = "https://rest.uniprot.org/uniprotkb/accessions"
  http_function = requests.get
  http_args = {'params': {'accessions': accessions}}

  return http_function(endpoint, **http_args)

def parse_response_uniprot(resp: dict):
    resp = resp.json()
    resp = resp["results"]

    output = {}
    for val in resp:
        acc = val['primaryAccession']
        species = val['organism']['scientificName']
        gene = val['genes'][0]['geneName']['value']
        seq = val['sequence']['value']
        seqlen = val['sequence']['length']
        output[acc] = {'organism':species, 'geneInfo':gene, 'sequenceInfo':seq, 'length': seqlen, 'type':'protein'}

    return output

def get_ensembl(ids: list):
    server = "https://rest.ensembl.org"
    ext = "/lookup/id"
    headers={ "Content-Type" : "application/json", "Accept" : "application/json"}
    string = '"'+'" , "'.join(ids)+'"' # Well, didnt know that double and single quotes can be such a difference....
    data = 	f'{{ "ids" : [{string}] }}'

    return requests.post(server+ext, headers=headers, data=data)

def parse_response_ensembl(resp: dict):
    resp = resp.json()
    output = {}

    for val in resp:
        acc = resp[val]['id']
        species = resp[val]['species']
        gene = resp[val]['display_name']
        assembly = resp[val]['assembly_name']
        type = resp[val]['biotype']
        chrom = resp[val]['seq_region_name']
        output[acc] = {'organism':species, 'assembly':assembly, 'geneInfo':gene,
                       'type':'nucleotide', 'biotype':type,
                       'chromosome': chrom}

    return output

def parse_list(ids: list, type):
      
    if type == 'Protein':
      
        return parse_response_uniprot(get_uniprot(ids))
      
    return parse_response_ensembl(get_ensembl(ids))


def seqkit_it(filename):
    seqkit = subprocess.run(("seqkit", "stats", filename, "-a"),
                            capture_output=True, text=True)
    return seqkit

def bioparser(filename):
    sequences = SeqIO.parse(filename, 'fasta')

    return sequences

def do(filename):
    seqkit = seqkit_it(filename)
    if seqkit.stderr:
        print(seqkit.stderr)
        return
    else:
        unipattern = re.compile(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}")
        enspattern = re.compile(r"ENS[A-Z]{1,5}[0-9]{11}") 
        ids = []


        seqkit_names, seqkit_stats = seqkit.stdout.strip().split('\n')[0].split(), seqkit.stdout.strip().split('\n')[1].split()
        type = seqkit_stats[2]

        sequences = bioparser(filename)
        for seq in sequences:
            seq_id = unipattern.search(seq.id).group(0) if type == 'Protein' else enspattern.search(seq.id).group(0)
            ids.append(seq_id)

        parsed = parse_list(ids, type)

        print('seqkit statistics:')
        print('___________________')
        print()
        for name, stat in zip(seqkit_names, seqkit_stats):
            print(name, ':', stat)

        print('___________________')
        print()

        if type == 'Protein':
            print('Unirot query results:')
        else:
            print('ENSEMBL lookup DB query results:')

        print('~~~~~~~~~~~~~~~')

        for x in parsed:
            print (x)
            for y in parsed[x]:
                print (y,':',parsed[x][y])
            print('~~~~~~~~~~~~~~~')

        return


if __name__ == '__main__':
    filename = 'hw_file2.fasta'
    do(filename)