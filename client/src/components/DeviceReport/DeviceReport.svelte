<script lang="ts">
    import GraphReport from "./GraphReport.svelte";
    const values: Array<{
        name: string,
        color: string,
        current: number,
        displayType: "string" | "graph"
    }> = [
        {
            name: "CPU",
            color: "#FF0000",
            current: 50,
            displayType: "graph"
        },
        {
            name: "GPU",
            color: "#00FF00",
            current: 50,
            displayType: "graph"
        },
        {
            name: "RAM",
            color: "#0000FF",
            current: 50,
            displayType: "graph"
        }
    ]
</script>

<main>
	<div class="wrapper">
        {#each values as value}
            <GraphReport name={value.name} current={value.current} color={value.color}/>
        {/each}
	</div>
</main>

<style>
	main {
        width: 100%;
        height: 100%;
        background-color: antiquewhite;
        position: relative;
	}

    .wrapper {
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: .3em;
    }
</style>