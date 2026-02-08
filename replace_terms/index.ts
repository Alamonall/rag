import { dirname, join } from "node:path";
import { replaceInMdFiles } from "./src/replace_in_md_files";
import { mkdir, writeFile, readFile } from 'node:fs/promises'
import { Term } from "./src/iterm";
import {generateUniqueNeutralTerm} from './src/generate_random_name';

// CLI entry point
async function main() {
  const termsMapFile = join(dirname(''),'../terms_map.json');
  const manualTermsMapFile = './docs/manual_terms.json';
  const docsFolder = '../knowledge_base_small';
  const outputFolder = join(dirname(''), '../knowledge_base_new');
  
  await mkdir(outputFolder, { recursive: true });

  console.log(`Scanning terms_map.json...`);

  let finalTerms: Term[] = [];
  const existingData = await readFile(manualTermsMapFile, 'utf-8');
  const terms: Term[] = JSON.parse(existingData);
  const generatedTerms = terms
    .filter((t: Term) => t.generated)
    .map((t: Term) => t.generated);

  terms.forEach(newTerm => {
    newTerm.generated = newTerm.generated ?? generateUniqueNeutralTerm(generatedTerms)
    finalTerms.push(newTerm);
  });
  
  await writeFile(termsMapFile, JSON.stringify(finalTerms, null, 2), 'utf-8');
  
  console.log(`Scanned terms_map.json with ${finalTerms.length} total terms`);

  const files = await replaceInMdFiles(docsFolder, finalTerms);

  let count = 1;
  for (const [fileName, content] of files) {
    console.log(`Writing #${count} file(${fileName}) to folder(${outputFolder})`);
    await writeFile(join(outputFolder, `doc_${count}.md`), content, 'utf-8');
    count++
  }
}

main();
