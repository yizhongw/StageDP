#!/usr/bin/env bash
#
# Runs Stanford CoreNLP.
# Simple uses for xml and plain text output to files are:
#    ./corenlp.sh -file filename
#    ./corenlp.sh -file filename -outputFormat text

scriptdir=`dirname $0`

# echo java -mx3g -cp \"$scriptdir/*\" edu.stanford.nlp.pipeline.StanfordCoreNLP $*

# $1 - path

PATH=$1
for FNAME in $PATH/*
do
    if [[ "$FNAME" == *text ]]
    then
        # /usr/bin/java -mx2g -cp "$scriptdir/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -tokenize.strictTreebank3 true -tokenize.latexQuotes false -tokenize.asciiQuotes false -tokenize.normalizeParentheses false -tokenize.normalizeOtherBrackets false -file $FNAME -outputDirectory $PATH -outputExtension '.xml'
        /usr/bin/java -mx2g -cp "$scriptdir/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -file $FNAME -outputDirectory $PATH -outputExtension '.xml'
        # /usr/bin/java -mx2g -cp "$scriptdir/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -file $FNAME
        # /bin/mv $(/usr/bin/basename $FNAME.xml) $PATH/
    fi
done
