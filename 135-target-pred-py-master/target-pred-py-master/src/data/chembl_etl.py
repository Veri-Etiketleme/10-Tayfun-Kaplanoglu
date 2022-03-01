"""
This module will download the SQLite version of the ChEMBL database if it
doesn't exist in data/, and use it to create the interim data sets.
"""

import csv
import os
import sqlite3
import tarfile
import urllib.request

OLD_QUERY = """
            SELECT 
                canonical_smiles, published_value, published_units, pref_name
            FROM 
                compound_structures as cs, activities as ac, assays as assays, target_dictionary as td
            WHERE 
                cs.molregno = ac.molregno 
            AND
                /* Endpoint is Ki*/
                ac.bao_endpoint = "BAO_0000192"
            AND
                ac.published_value IS NOT NULL
            AND
                ac.assay_id = assays.assay_id
            AND
                assays.tid = td.tid
            AND
                td.pref_name != "Unchecked"
            AND
                td.organism = "Homo sapiens"
            """

SWISS_QUERY = """
                SELECT
                    canonical_smiles, standard_value, standard_units, pref_name
                FROM
                    compound_structures as cs, target_dictionary as td, assays, activities, compound_properties
                WHERE
                    compound_properties.molregno = cs.molregno
                AND
                    cs.molregno = activities.molregno
                AND
                    activities.assay_id = assays.assay_id
                AND 
                    assays.tid = td.tid
                AND
                    assays.assay_organism = "Homo sapiens"
                AND
                    assays.assay_type = "B"
                AND
                    activities.type IN ("Ki","Kd","IC50","EC50")
                AND
                    activities.standard_units IN ("mM", "uM","nM")
                AND
                    activities.standard_relation in ("=", "<", "<=")
                AND
                    compound_properties.heavy_atoms < 80
                AND
                    td.target_type IN ("SINGLE PROTEIN", "PROTEIN COMPLEX")
              """


class ChEMBL_SQLite:  # pylint: disable=invalid-name
    """ChEMBL data http://www.ebi.ac.uk/chembl version chembl_24_1.

    Args:
        path (string, optional): Where the data will be downloaded. Defaults to
            the `data` directory of code.
    """
    url = "ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_25/"
    filename = "chembl_25_sqlite.tar.gz"
    dbpath = "chembl_25/chembl_25_sqlite/chembl_25.db"
    csvfilename = "../interim/smiles_to_activity.csv"

    def __init__(self, path="../../data/external/"):
        self.path = path

    def get_raw_data(self):
        """Will create the raw data if it does not already exist."""
        if not os.path.isfile(self.path + self.csvfilename):
            self._write_raw_data(query=SWISS_QUERY)

    def db_connect(self):
        """Returns a connection to the ChEMBL database,
        will download if it does not exist.

        Returns:
            conn (sqlite3.Connection): The connection to the database
        """
        if not os.path.isfile(self.path + self.dbpath):
            self._download()

        conn = sqlite3.connect(self.path + self.dbpath)
        return conn

    def _write_raw_data(self, query):
        """This runs the query to get our data from the database
        For now this query returns good-enough data to do a quick analysis,
        this is not the final query.
        """

        conn = self.db_connect()
        print("Running SQL query")
        cur = conn.execute(query)
        headers = [x[0] for x in cur.description]
        data_table = cur.fetchall()

        with open(self.path + self.csvfilename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data_table)

    def _download(self):
        """Downloads the ChEMBL database if it doesn't exist"""
        if not os.path.isfile(self.path + self.dbpath):
            delete_tar = False

            if not os.path.isfile(self.path + self.filename):
                delete_tar = True
                print("Downloading ChEMBL database")
                urllib.request.urlretrieve(self.url + self.filename,
                                           self.path + self.filename)

            print("Extracting tarfile")
            tarfile.open(self.path + self.filename).extractall(path=self.path)
            if delete_tar:
                os.remove(self.path + self.filename)
