# Known Issues

This section will show all known differences from the [Neuro API](https://github.com/VedalAI/neuro-game-sdk/blob/main/API/SPECIFICATION.md), if you spot a new one, let me know by opening an issue or by making a PR.

## General Differences

- The api does not use the `game` value ever

## Command Differences

`action/result`
- success boolean is ignored and is always treated as successful
- if an actions/force is present, it will always be discarded once an action is performed regardless of if it was successful.

