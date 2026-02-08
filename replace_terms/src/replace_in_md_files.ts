import { stat, readFile, writeFile } from 'node:fs/promises';
import  { dirname, join } from 'node:path';
import { glob } from 'glob';
import { Term } from './iterm';


function addUnderscoreToSecondPlace(string: string) {
  return `${string.slice(0, 1)}_${string.slice(1)}`
}

function findByWord(terms: Term[], words: string[]) {
  for(let word of words) {
    const found = terms.find((t: Term) => t.original.toLowerCase() === word.toLowerCase());

    if(found) {
      return found
    }
  }
}

export async function replaceInMdFiles(
  folderPath: string,
  terms: Term[]
) {
  const mdFiles = await glob(join(folderPath, '**/*.md'));
  console.log(`Scanning ${mdFiles.length} Markdown files...`);
  const newContentFiles = new Map<string, string>();

  // const lostOriginalsFilePath = join(dirname(''),'./docs/lost.json')
  // const isExists = await stat(lostOriginalsFilePath)
  //   .then(() => true)
  //   .catch(() => false);
  // if(!isExists) {
  //   await writeFile(lostOriginalsFilePath, JSON.stringify([]), 'utf-8')
  // }
  // const lostOriginalsFileContentRaw = await readFile(lostOriginalsFilePath, 'utf-8');
  // const lostOriginalsFileContent = JSON.parse(lostOriginalsFileContentRaw) as { word: string; file: string }[]

  let docCount = 0
  for (const filePath of mdFiles) {
      const content = await readFile(filePath, 'utf-8');
      const updated = terms.reduce((acc: string, term: Term) => {
        const termToReplace = new RegExp(addUnderscoreToSecondPlace(term.original), 'gi'); 

        if(term.stems) {
          for (const stem of term.stems) {
            const stemToReplace = new RegExp(addUnderscoreToSecondPlace(stem), 'gi'); 
            acc = acc.replaceAll(
              stemToReplace,
              term.generated
            );
          }
        }

        // if(acc.includes(`${term.original} `)
        //   && acc.includes(` ${term.original}`)
        //   && !lostOriginalsFileContent.some(t => t.word.toLowerCase() === term.original.toLowerCase())) {
        //     lostOriginalsFileContent.push({ word: term.original, file: filePath })
        // }

        return acc.replaceAll(
          termToReplace,
          term.generated
        );
      }, content)

      const fileName = filePath.split('/').pop()?.split('.')[0];
      if(!fileName) {
        throw new Error(`File not found: ${filePath}`)
      }

      // await writeFile(lostOriginalsFilePath, JSON.stringify(lostOriginalsFileContent), 'utf-8');
      
      // const targetTerm = findByWord(terms, [fileName, ...fileName.split(" ")])

      // if(!targetTerm) {
      //   throw new Error(`Filename(${fileName}) has not matching any term by words(${fileName.split(" ")})`)
      // }
      newContentFiles.set(`doc_${docCount}` , updated);
      docCount++;
  }
  
  return newContentFiles;
}
