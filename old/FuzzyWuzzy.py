# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:02:47 2019

@author: cfarris
"""

from fuzzywuzzy import fuzz
#from fuzzywuzzy import process
from collections import OrderedDict

searchTerm = 'CUST 326546841644 MEDIARENA CORTLAND BALANCE CONTROL DETAIL REF: 220 CUST:'
searchTerms = searchTerm.upper().split()
#print(searchTerms)
bankLoanIssuers = ['Allison Transmission Holdings, Inc.','Allison Transmission Holdings, Inc.','MEDIARENA ACQUISITION B.V.','MEDIARENA ACQUISITION B.V.','Sophia L.P.','MGM Growth Properties Operating Partnership LP','MGM Growth Properties Operating Partnership LP','MGM Growth Properties Operating Partnership LP','Micron Technology, Inc.','Kronos Incorporated','Kronos Incorporated','SERTA SIMMONS HOLDINGS, LLC','Ineos U.S. Finance LLC','Ineos U.S. Finance LLC','Ineos U.S. Finance LLC','Ineos U.S. Finance LLC','Ineos U.S. Finance LLC','Valeant Pharmaceuticals International, Inc.','Valeant Pharmaceuticals International, Inc.','HCA Healthcare, Inc.','Sophia L.P.','EFS COGEN HOLDINGS I LLC','EMG UTICA, LLC','Sensata Technologies Finance Company, LLC','BRONCO MIDSTREAM FUNDING','Allison Transmission Holdings, Inc.','MEDIARENA ACQUISITION B.V.','ARBOR PHARMACEUTICALS, INC.','PHILLIPS PET FOOD &amp; SUPPLIES','CDW LLC','GULF FINANCE, LLC','MGM Growth Properties Operating Partnership LP','Micron Technology, Inc.','Kronos Incorporated','The ServiceMaster Company, LLC','SERTA SIMMONS HOLDINGS, LLC','Astoria Energy','Valeant Pharmaceuticals International, Inc.','Valeant Pharmaceuticals International, Inc.','SERTA SIMMONS HOLDINGS, LLC','Virgin Media Bristol LLC','Valeant Pharmaceuticals International, Inc.','BRONCO MIDSTREAM FUNDING','Sophia L.P.','EFS COGEN HOLDINGS I LLC','EMG UTICA, LLC','Zebra Technologies Corp','Zebra Technologies Corp','Advanced Disposal Services, Inc. (fka ADS Waste Holdings, Inc.)','Zebra Technologies Corp','Advanced Disposal Services, Inc. (fka ADS Waste Holdings, Inc.)','Prestige Brands, Inc.','Prestige Brands, Inc.','Advanced Disposal Services, Inc. (fka ADS Waste Holdings, Inc.)','Advanced Disposal Services, Inc. (fka ADS Waste Holdings, Inc.)','Advanced Disposal Services, Inc. (fka ADS Waste Holdings, Inc.)','Advanced Disposal Services, Inc. (fka ADS Waste Holdings, Inc.)','Select Medical Corporation','Sensata Technologies Finance Company, LLC','Level 3 Financing, Inc.','IQVIA, Inc.','Zebra Technol']
bankLoanIssuers = set(bankLoanIssuers)
bankLoanIssuers = list(bankLoanIssuers)
for x in range(len(bankLoanIssuers) - 1):
    bankLoanIssuers[x] = bankLoanIssuers[x].upper()

searchDict = OrderedDict()
for x in range(len(searchTerms) - 1):
    for y in range(len(bankLoanIssuers) - 1):
        index = str(searchTerms[x] + ' || ' + bankLoanIssuers[y])
        ratio = fuzz.partial_ratio(searchTerms[x], bankLoanIssuers[y])
        if ratio > 0:
            searchDict[index] = ratio

searchResults = {k: v for k, v in sorted(searchDict.items(), key = lambda x: x[1], reverse = True)}
searchResults = list(searchResults.items())
print(searchResults[0])





