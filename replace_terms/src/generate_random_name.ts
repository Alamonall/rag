
/**
 * Генератор фентезийных имен из одного слова
 * Избегает имен из "Властелина Колец"
 */

function generateFantasyName(): string {
  // Список имен из LOTR, которые нужно избегать
  const lotrNames = new Set([
    // Персонажи
    'Aragorn', 'Arwen', 'Bilbo', 'Boromir', 'Celeborn', 'Denethor',
    'Eowyn', 'Faramir', 'Frodo', 'Galadriel', 'Gandalf', 'Gimli',
    'Gollum', 'Legolas', 'Merry', 'Pippin', 'Samwise', 'Saruman',
    'Sauron', 'Theoden', 'Treebeard', 'Wormtongue',
    
    // Места
    'Bree', 'Gondor', 'Hobbiton', 'Isengard', 'Lothlorien', 'Mirkwood',
    'Mordor', 'Rivendell', 'Rohan', 'Shire', 'Weathertop',
    
    // Расы/народы
    'Balrog', 'Ent', 'Hobbit', 'Maiar', 'Nazgul', 'Orc', 'Uruk',
    
    // Артефакты
    'Anduril', 'Mithril', 'Narsil', 'Palantir', 'Ring', 'Sting'
  ]);

  // Наши уникальные компоненты
  const components = {
    // Слоги для эльфийских имен (не LOTR)
    elvenStart: [
      'Ae', 'Al', 'Ca', 'Da', 'Ea', 'Fa', 'Ga', 'Ha', 'Ia', 'Ja',
      'Ka', 'La', 'Ma', 'Na', 'Oa', 'Pa', 'Qa', 'Ra', 'Sa', 'Ta',
      'Ua', 'Va', 'Wa', 'Xa', 'Ya', 'Za'
    ],
    elvenMid: [
      'ran', 'len', 'mir', 'nor', 'sil', 'tar', 'ven', 'wyn', 'xel', 'yor',
      'bor', 'cor', 'dor', 'for', 'gor', 'hor', 'jor', 'kor', 'lor', 'mor'
    ],
    elvenEnd: [
      'ion', 'ath', 'eth', 'ith', 'oth', 'uth', 'ael', 'iel', 'uel',
      'orn', 'arn', 'ern', 'irn', 'orn', 'urn', 'ari', 'eri', 'ori',
      'aro', 'ero', 'oro', 'uro'
    ],

    // Гномьи имена (не LOTR)
    dwarfStart: [
      'Bar', 'Bur', 'Dar', 'Dur', 'Far', 'Gar', 'Har', 'Kar', 'Mar', 'Nar',
      'Par', 'Rar', 'Sar', 'Tar', 'Var', 'War', 'Xar', 'Yar', 'Zar'
    ],
    dwarfEnd: [
      'ak', 'ek', 'ik', 'ok', 'uk', 'ag', 'eg', 'ig', 'og', 'ug',
      'an', 'en', 'in', 'on', 'un', 'ath', 'eth', 'ith', 'oth'
    ],

    // Человеческие имена
    humanStart: [
      'Ald', 'Bran', 'Ced', 'Dag', 'Ed', 'Fred', 'Greg', 'Hugh', 'Ivan',
      'Jack', 'Kael', 'Leo', 'Mark', 'Ned', 'Owen', 'Paul', 'Quinn',
      'Rex', 'Sean', 'Tom', 'Ulf', 'Vik', 'Will', 'Xan', 'York', 'Zane'
    ],
    humanEnd: [
      'ric', 'win', 'hard', 'son', 'ward', 'bert', 'mond', 'fred', 'nard',
      'ton', 'ville', 'field', 'wood', 'stone', 'heart', 'blade', 'shield'
    ],

    // Магические существа
    creatureStart: [
      'Chry', 'Ember', 'Glimmer', 'Ignis', 'Lumin', 'Nova', 'Onyx', 'Pyre',
      'Quartz', 'Ruby', 'Sapphire', 'Topaz', 'Umber', 'Vivid', 'Wisp', 'Xeno'
    ],
    creatureEnd: [
      'drake', 'fang', 'claw', 'scale', 'wing', 'tail', 'horn', 'mane',
      'sight', 'breath', 'fire', 'frost', 'storm', 'rage', 'grace'
    ],

    // Артефакты
    artifactStart: [
      'Astral', 'Celestial', 'Divine', 'Ethereal', 'Forgotten', 'Hidden',
      'Infernal', 'Jade', 'Karmic', 'Lost', 'Mystic', 'Noble', 'Obsidian',
      'Primal', 'Quantum', 'Radiant', 'Spectral', 'Titan', 'Umbral', 'Void'
    ],
    artifactEnd: [
      'blade', 'staff', 'orb', 'shield', 'crown', 'ring', 'amulet', 'seal',
      'key', 'scroll', 'tome', 'mirror', 'hourglass', 'phylactery', 'core'
    ]
  };

  // Функция проверки на LOTR
  const isLotrName = (name: string): boolean => {
    return lotrNames.has(name);
  };

  // Генерируем пока не получим уникальное имя
  let name: string;
  let attempts = 0;
  
  do {
    attempts++;
    if (attempts > 100) {
      return 'Mysterious'; // fallback
    }

    const type = Math.floor(Math.random() * 6);
    
    switch (type) {
      case 0: // Эльфийское имя
        const elvenStart = components.elvenStart[Math.floor(Math.random() * components.elvenStart.length)];
        const elvenMid = Math.random() > 0.5 ? components.elvenMid[Math.floor(Math.random() * components.elvenMid.length)] : '';
        const elvenEnd = components.elvenEnd[Math.floor(Math.random() * components.elvenEnd.length)];
        name = elvenStart + elvenMid + elvenEnd;
        break;

      case 1: // Гномье имя
        const dwarfStart = components.dwarfStart[Math.floor(Math.random() * components.dwarfStart.length)];
        const dwarfEnd = components.dwarfEnd[Math.floor(Math.random() * components.dwarfEnd.length)];
        name = dwarfStart + dwarfEnd;
        break;

      case 2: // Человеческое имя
        const humanStart = components.humanStart[Math.floor(Math.random() * components.humanStart.length)];
        const humanEnd = components.humanEnd[Math.floor(Math.random() * components.humanEnd.length)];
        name = humanStart + humanEnd;
        break;

      case 3: // Существо
        const creatureStart = components.creatureStart[Math.floor(Math.random() * components.creatureStart.length)];
        const creatureEnd = components.creatureEnd[Math.floor(Math.random() * components.creatureEnd.length)];
        name = creatureStart + creatureEnd;
        break;

      case 4: // Артефакт
        const artifactStart = components.artifactStart[Math.floor(Math.random() * components.artifactStart.length)];
        const artifactEnd = components.artifactEnd[Math.floor(Math.random() * components.artifactEnd.length)];
        name = artifactStart + artifactEnd;
        break;

      case 5: // Смешанное
        const allParts = [
          ...components.elvenStart, ...components.elvenMid, ...components.elvenEnd,
          ...components.dwarfStart, ...components.dwarfEnd,
          ...components.humanStart, ...components.humanEnd
        ];
        const syllableCount = Math.floor(Math.random() * 3) + 2;
        name = '';
        for (let i = 0; i < syllableCount; i++) {
          name += allParts[Math.floor(Math.random() * allParts.length)];
        }
        break;

      default:
        name = 'Mysterious';
    }

    // Капитализируем первую букву
    name = name.charAt(0).toUpperCase() + name.slice(1).toLowerCase();
    
  } while (isLotrName(name));

  return name;
}

// Generate unique neutral replacement
export function generateUniqueNeutralTerm(existingTerms: string[]): string {
  let attempts = 0;
  const maxAttempts = 100;
  
  while (attempts < maxAttempts) {
    const newTerm = generateFantasyName();
    if (!existingTerms.includes(newTerm)) {
      return newTerm;
    }
    attempts++;
  }
  return `term${Date.now()}`;
}
