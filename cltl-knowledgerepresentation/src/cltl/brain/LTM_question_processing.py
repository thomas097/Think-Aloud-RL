from rdflib import Literal


######################################### Helpers for question processing #########################################

def create_query(self, utterance):
    empty = ['', Literal(''), 'unknown', 'none']

    # Query subject
    if utterance['subject']['label'] is None or utterance['subject']['label'].lower() in empty:
        query = """
                   SELECT distinct ?slabel ?authorlabel ?certaintyValue ?polarityValue ?sentimentValue ?emotionValue ?temporalValue
                           WHERE { 
                               ?s n2mu:%s ?o . 
                               ?s rdfs:label ?slabel . 
                               ?o rdfs:label '%s' .  
                               GRAPH ?g {
                                   ?s n2mu:%s ?o . 
                               } . 
                               ?g gaf:denotedBy ?m . 
                               ?m grasp:wasAttributedTo ?author . 
                               ?author rdfs:label ?authorlabel .
                               ?m grasp:hasAttribution ?att .
                               
                               OPTIONAL {
                               ?att rdf:value ?certainty .
                               ?certainty rdf:type graspf:CertaintyValue .
                               ?certainty rdfs:label ?certaintyValue .
                               }
                               OPTIONAL {
                               ?att rdf:value ?polarity .
                               ?polarity rdf:type graspf:PolarityValue .
                               ?polarity rdfs:label ?polarityValue .
                               }
                               OPTIONAL {
                               ?att rdf:value ?temporal .
                               ?temporal rdf:type graspf:TemporalValue .
                               ?temporal rdfs:label ?temporalValue .
                               }
                               OPTIONAL {
                               ?att rdf:value ?emotion .
                               ?emotion rdf:type graspe:EmotionValue .
                               ?emotion rdfs:label ?emotionValue .
                               }
                               OPTIONAL {
                               ?att rdf:value ?sentiment .
                               ?sentiment rdf:type grasps:SentimentValue .
                               ?sentiment rdfs:label ?sentimentValue .
                               }
                           }
                   """ % (utterance['predicate']['label'],
                          utterance['object']['label'],
                          utterance['predicate']['label'])

    # Query complement
    elif utterance['object']['label'] is None or utterance['object']['label'].lower() in empty:
        query = """
                   SELECT distinct ?olabel ?authorlabel ?certaintyValue ?polarityValue ?sentimentValue ?emotionValue ?temporalValue
                           WHERE { 
                               ?s n2mu:%s ?o .   
                               ?s rdfs:label '%s' .  
                               ?o rdfs:label ?olabel .  
                               GRAPH ?g {
                                   ?s n2mu:%s ?o . 
                               } . 
                               ?g gaf:denotedBy ?m . 
                               ?m grasp:wasAttributedTo ?author . 
                               ?author rdfs:label ?authorlabel .
                               ?m grasp:hasAttribution ?att .
                               
                               OPTIONAL {
                               ?att rdf:value ?certainty .
                               ?certainty rdf:type graspf:CertaintyValue .
                               ?certainty rdfs:label ?certaintyValue .
                               }
                               OPTIONAL {
                               ?att rdf:value ?polarity .
                               ?polarity rdf:type graspf:PolarityValue .
                               ?polarity rdfs:label ?polarityValue .
                               }
                               OPTIONAL {
                               ?att rdf:value ?temporal .
                               ?temporal rdf:type graspf:TemporalValue .
                               ?temporal rdfs:label ?temporalValue .
                               }
                               OPTIONAL {
                               ?att rdf:value ?emotion .
                               ?emotion rdf:type graspe:EmotionValue .
                               ?emotion rdfs:label ?emotionValue .
                               }
                               OPTIONAL {
                               ?att rdf:value ?sentiment .
                               ?sentiment rdf:type grasps:SentimentValue .
                               ?sentiment rdfs:label ?sentimentValue .
                               }
                           }
                   """ % (utterance['predicate']['label'],
                          utterance['subject']['label'],
                          utterance['predicate']['label'])

    # Query existence
    else:
        query = """
                   SELECT distinct ?authorlabel ?certaintyValue ?polarityValue ?sentimentValue ?emotionValue ?temporalValue
                           WHERE { 
                               ?s n2mu:%s ?o .   
                               ?s rdfs:label '%s' .  
                               ?o rdfs:label '%s' .  
                               GRAPH ?g {
                                   ?s n2mu:%s ?o . 
                               } . 
                               ?g gaf:denotedBy ?m . 
                               ?m grasp:wasAttributedTo ?author . 
                               ?author rdfs:label ?authorlabel .
                               ?m grasp:hasAttribution ?att .
                               
                               OPTIONAL {
                               ?att rdf:value ?certainty .
                               ?certainty rdf:type graspf:CertaintyValue .
                               ?certainty rdfs:label ?certaintyValue .
                               }
                               OPTIONAL {
                               ?att rdf:value ?polarity .
                               ?polarity rdf:type graspf:PolarityValue .
                               ?polarity rdfs:label ?polarityValue .
                               }
                               OPTIONAL {
                               ?att rdf:value ?temporal .
                               ?temporal rdf:type graspf:TemporalValue .
                               ?temporal rdfs:label ?temporalValue .
                               }
                               OPTIONAL {
                               ?att rdf:value ?emotion .
                               ?emotion rdf:type graspe:EmotionValue .
                               ?emotion rdfs:label ?emotionValue .
                               }
                               OPTIONAL {
                               ?att rdf:value ?sentiment .
                               ?sentiment rdf:type grasps:SentimentValue .
                               ?sentiment rdfs:label ?sentimentValue .
                               }
                           }
                   """ % (utterance['predicate']['label'],
                          utterance['subject']['label'],
                          utterance['object']['label'],
                          utterance['predicate']['label'])

    query = self.query_prefixes + query

    self._log.info(f"Triple in question: {utterance['triple']}")

    return query
