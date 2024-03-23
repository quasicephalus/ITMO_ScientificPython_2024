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
        gene = val['genes']
        seq = val['sequence']
        output[acc] = {'organism':species, 'geneInfo':gene, 'sequenceInfo':seq, 'type':'protein'}

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
        trans = resp[val]['canonical_transcript']
        chrom = resp[val]['seq_region_name']
        output[acc] = {'organism':species, 'assembly':assembly, 'geneInfo':gene,
                       'type':'nucleotide', 'biotype':type, 'transcript':trans,
                       'chromosome': chrom}

    return output

def parse_list(ids: list):
    unipattern = re.compile(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}")
    # ENSEMBLE id has "ENS" in the start, then either 3 letters organism ID and 1-2 letters feature ID,
    # or just feature ID in case og Human, 
    # and at last 11-digit identifier
    enspattern = re.compile(r"ENS[A-Z]{1,5}[0-9]{11}") 
    db = None
    if all(unipattern.match(id) for id in ids):
        db = 'uniprot'
    elif all(enspattern.match(id) for id in ids):
        db = 'ensembl'
    else:
      
        return 'none of IDs are uniquely from ENSEMBL or Uniprot'
      
    if db == 'ensembl':
      
        return parse_response_ensembl(get_ensembl(ids))
      
    return parse_response_uniprot(get_uniprot(ids))
    
if __name__ == '__main__':

    ens_ids = ['ENSMUSG00000041147', 'ENSG00000139618']
    uni_ids = ['P11473', 'Q91XI3']

    print(parse_list(ens_ids))
    print(parse_list(uni_ids))
