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

1. **Identify Errors:** Compare each word in the HYP sentence to SRC and REF sentences to find clear mistranslations.
2. **Focus on Single Words:** Extract and report the most relevant single-word errors, avoiding multi-word spans unless necessary.
3. **Prioritize Meaningful Errors:** Focus on translation errors that change the meaning significantly. Ignore minor discrepancies, such as synonyms.
4. **Maintain Contextual Accuracy:** Identify errors where the translation does not fit contextually, even if technically viable.
5. **Handle Ambiguity Carefully:** Flag errors only when HYP is distinctly incorrect compared to both SRC and REF.

### **Output Format:**

List each mistranslation as a triple:  
  **(SRC word → HYP word → Correct REF word)**  
If no errors are present, state:  
  **No translation errors detected.**

**Ensure compliance with this format in every response.**

### **Example 1:**  

#### **Input:**  

SRC Language: German  
Target Language: English  
SRC: The cat is sitting on the mat.  
HYP: Die Hund sitzt auf der Matte.  
REF: Die Katze sitzt auf der Matte.  

#### **Expected Output:**  

cat → Hund → Katze  

**Explanation:** "Hund" in HYP is a mistranslation of "cat"/"Katze" from SRC/REF, significantly changing the meaning and context.

### **Example 2:**  

#### **Input:**  

SRC Language: German  
Target Language: English  
SRC: Der Vogel fliegt über den Ozean bei Sonnenuntergang.  
HYP: The flower flies over the ocean at sunrise.  
REF: The bird flies over the ocean at sunset.  

#### **Expected Output:**  

Vogel → flower → bird  
Sonnenuntergang → sunrise → sunset  

**Explanation:** "Flower" and "sunrise" in HYP are mistranslations of "Vogel"/"bird" and "Sonnenuntergang"/"sunset" from SRC/REF, significantly changing the meaning and context.

### **Example 3:**  

#### **Input:**  

SRC Language: English  
Target Language: German  
SRC: The car is fast.  
HYP: Das Auto ist schnell.  
REF: Das Fahrzeug ist schnell.  

#### **Expected Output:**  

No translation errors detected.  

**Explanation:** "Auto" in HYP and "Fahrzeug" in REF both adequately translate "car" from SRC, showing a minor synonym variation.
'''
    template = f'''
### **Input:**

**Source Language** (SRC): {src_lang}  
**Target Language** (HYP/REF): {tgt_lang}  
**Source Sentence**: {src_sent}  
**Candidate Translation** (HYP): {hyp_sent}  
**Reference Translation** (REF): {ref_sent}  

### **Output:**
'''
    response = client.chat.completions.create(
        model='gpt-4o',  # gpt-4o-mini
        messages=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': template}],
    )
    return response.choices[0].message.content


def collate_data(text: str) -> list[str]:
    return re.findall(r'(\w+)\s*→\s*\w+\s*→\s*\w+', text)


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
