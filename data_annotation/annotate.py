import argparse
import re
import sys

from openai import OpenAI
from tqdm import tqdm

client = OpenAI()
src_lang, tgt_lang = 'English', 'German'


def annotate_data(src_sent: str, ref_sent: str, hyp_sent: str) -> str | None:
    prompt = '''
### **Task:**

Identify and list all single-word translation errors in a machine-translated sentence.

### **Instructions:**

1. **Identify Mistranslations:** Compare each word in the candidate sentence with the source and reference sentences to find clear translation errors.
2. **Focus on Single-Word Errors:** Report errors at the word level. If a multi-word phrase is mistranslated, extract the most significant word.
3. **Prioritize Meaningful Errors:** Report only errors that significantly change meaning. Ignore acceptable variations, such as near-synonyms.
4. **Ensure Contextual Accuracy:** Identify words that, while potentially valid in isolation, do not fit the intended meaning in context.
5. **Handle Ambiguity Carefully:** Mark an error only if the candidate word is demonstrably incorrect when compared to the source and reference.

**Strict Adherence Required:** Always follow the output format exactly, and include detailed explanations that refer back to the instructions.

### **Output Format:**

List each mistranslation as a triple in the following format:
**(source word → candidate word → reference word)**
If there are no translation errors, output:
**No translation errors detected.**

### **Example 1:**

#### **Input:**

**Source Language:** English
**Target Language:** German
**Source Sentence:** The scientist presented a groundbreaking discovery.
**Candidate Translation:** Der Wissenschaftler präsentierte eine einfache Entdeckung.
**Reference Translation:** Der Wissenschaftler präsentierte eine bahnbrechende Entdeckung.

#### **Expected Output:**  

groundbreaking → einfache → bahnbrechende  

**Explanation:** "einfache" (simple) is a mistranslation of "groundbreaking", which significantly changes the meaning of the sentence. This violates the **Prioritize Meaningful Errors** rule because it downplays the significance of the discovery.

### **Example 2:**  

#### **Input:**  

**Source Language:** English  
**Target Language:** German  
**Source Sentence:** The bird flies over the ocean towards the setting sun.
**Candidate Translation:** Der Blume fliegt über den Ozean in Richtung des verschwindenden Mondes.
**Reference Translation:** Der Vogel fliegt über den Ozean in Richtung der untergehenden Sonne.

#### **Expected Output:**  

bird → Blume → Vogel
sun → Mondes → Sonne  

**Explanation:** "Blume" (flower) is a mistranslation of "bird". This violates the **Ensure Contextual Accuracy** rule because flowers cannot fly. "Mondes" (moon) is a mistranslation of "sun", which changes the intended imagery. Since the key error is the celestial body itself, "verschwindenden Mondes" (disappearing moon) is reduced to "Mondes" (moon) and "untergehenden Sonne" (setting sun) is reduced to "Sonne" (sun), following the **Focus on Single-Word Errors** rule.

### **Example 3:**  

#### **Input:**  

**Source Language:** English  
**Target Language:** German  
**Source Sentence:** The lawyer prepared the contract.
**Candidate Translation:** Der Anwalt setzte den Vertrag auf.
**Reference Translation:** Der Anwalt bereitete den Vertrag vor.

#### **Expected Output:**  

No translation errors detected.  

**Explanation:** "setzte auf" and "bereitete vor" are both valid ways to express "prepared" in this context. Since the meaning remains intact, this follows the **Handle Ambiguity Carefully** rule, and no error is reported.
'''
    template = f'''
### **Input:**

**Source Language:** {src_lang}
**Target Language:** {tgt_lang}
**Source Sentence:** {src_sent}
**Candidate Translation:** {hyp_sent}
**Reference Translation:** {ref_sent}

### **Output:**
'''
    response = client.chat.completions.create(
        model='gpt-4o',  # gpt-4o-mini
        messages=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': template}],
    )
    return response.choices[0].message.content


def collate_data(text: str) -> list[str]:
    matches = re.findall(r"([\w'-]+)\s*→\s*([\w'-]+)\s*→\s*([\w'-]+)", text)
    return [src_word for src_word, hyp_word, ref_word in matches if hyp_word != ref_word]


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand')
    annotate = subparsers.add_parser('annotate')
    annotate.add_argument('source', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    annotate.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    collate = subparsers.add_parser('collate')
    collate.add_argument('source', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    collate.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    args = parser.parse_args()

    match args.subcommand:
        case 'annotate':
            for line in tqdm(args.source.readlines()):
                content = annotate_data(*line.strip().split('\t'))
                args.output.write(content.replace('\n', '\t') + '\n')
        case 'collate':
            for line in tqdm(args.source.readlines()):
                args.output.write(', '.join(collate_data(line)) + '\n')


if __name__ == '__main__':
    main()
