use std::fmt;

#[derive(Debug,Clone,Copy,PartialEq,Eq)]

pub struct Card(pub u8);

impl Card {
    pub fn new(id:u8) -> Self {
        Card(id)
    }

    pub fn suit(&self) -> u8 {
        self.0 / 13
    }

    pub fn rank(&self) -> u8 {
        self.0 % 13
    }

    pub fn neighbor(&self,offset:i8) -> Option<Card>{
        let r = self.rank() as i8 + offset;

        if r >= 0 && r < 13 {
            Some(Card((self.suit() * 13 ) + r as u8))
        } else {
            None
        }
    }
}

impl fmt::Display for Card {
    fn fmt (&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        let suits = ["♦","♥","♣","♠"];
        let ranks = ["A","2","3","4","5","6","7",
           "8","9","10","J","Q","K"];
           write!(f,"{}{}",suits[self.suit() as usize],ranks[self.rank() as usize])
    }
}