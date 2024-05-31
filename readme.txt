This program is designed to train a model that can play the
coin flipping game at https://primerlearning.org/
(or at least predict the best move)

The aim of the game is to identify cheaters and fair players
based on their flips of a coin.
Cheaters have a 75% chance of getting heads.
The user can decide to either:
Flip a coin
Flip 5 coins
Label as fair
Label as cheater

To simplify the model I grouped both flip a coin and 5 coins to 
"require more data to be sure".
So based on the results of heads vs tails, the model classifies
the data to either cheater or fair player. If the percent likelihood
is smaller than specified, then it returns that it needs more data.

This could quite easily be done using hypothesis testing,
this program actually performs worse than hypothesis testing 
due to it not being able to extrapolate data very well to more samples
but I wanted to practice training a model and it's more fun.

There are some things which are intentionally not super optimised
for learning purposes. Such as writing to csv then reading the same file
to practice reading a csv with genfromtxt