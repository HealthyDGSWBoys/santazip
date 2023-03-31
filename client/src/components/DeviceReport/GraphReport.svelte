<script lang="ts">
    export let color: string
    export let current: number
    export let name: string
</script>

<main>
    <h6>
        {name}
    </h6>
    <div class="progress-container">
        <div 
            class="progress-bar" 
            style="width: {current}%; background-color: {color}">
        </div>
    </div>
    
</main>

<style>
	main {
        width: 100%;
        height: 2em;
        background-color: #aaa;
        display: flex;
        align-items: center;
        padding: .1em;
        position: relative;
	}

    .progress-container {
        width: 80%;
        height: 60%;
        position: absolute;
        right: .2em;
        background-color: aliceblue;
    }
    .progress-bar {
        transition: 2000ms;
        height: 100%;
        position: absolute;
        left: 0;
    }
</style>