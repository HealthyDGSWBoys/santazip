<script lang="ts">
        import type { learningItems } from './learningItems'
        let learningItems: learningItems = {
                sequential: new Array(),
                processer: new Array(),
                optimizer: "Adam",
                loss: 'SparseCategoricalCrossentropy',
                metrics: new Array(),
                epochs: 1024,
                batch: 64
        }
</script>

<main>
</main>

<style>
	main {
        width: 100%;
        height: 100%;
        background-color: antiquewhite;
        position: relative;
	}
</style>